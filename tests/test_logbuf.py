"""Tests for app/logbuf.py — in-memory log capture served by GET /api/logs."""
import logging
import sys

import pytest

from app.logbuf import LEVELS, LogRing, RingHandler, TeeStream


@pytest.fixture
def ring():
    return LogRing(capacity=5)


class TestLogRing:
    def test_seq_is_monotonic_so_clients_can_poll_without_gaps(self, ring):
        for i in range(3):
            ring.add("INFO", "t", f"line {i}")
        seqs = [e["seq"] for e in ring.entries()["entries"]]
        assert seqs == [1, 2, 3]

    def test_evicts_oldest_and_says_so(self, ring):
        for i in range(8):
            ring.add("INFO", "t", f"line {i}")
        snap = ring.entries()
        assert len(snap["entries"]) == 5
        assert snap["entries"][0]["message"] == "line 3"
        assert snap["dropped"] == 3, "a silent gap looks like a quiet system"

    def test_seq_keeps_counting_past_eviction(self, ring):
        for i in range(8):
            ring.add("INFO", "t", f"line {i}")
        assert ring.entries()["next_seq"] == 8

    def test_since_seq_returns_only_what_is_new(self, ring):
        for i in range(4):
            ring.add("INFO", "t", f"line {i}")
        snap = ring.entries(since_seq=2)
        assert [e["message"] for e in snap["entries"]] == ["line 2", "line 3"]

    def test_level_filter_is_a_minimum_not_an_exact_match(self, ring):
        ring.add("INFO", "t", "chatter")
        ring.add("WARNING", "t", "hmm")
        ring.add("ERROR", "t", "bang")
        msgs = [e["message"] for e in ring.entries(level="WARNING")["entries"]]
        assert msgs == ["hmm", "bang"]

    def test_unknown_level_filter_shows_everything(self, ring):
        """A typo in the filter must not look like an empty log."""
        ring.add("INFO", "t", "chatter")
        assert len(ring.entries(level="LOUD")["entries"]) == 1

    def test_limit_keeps_the_newest(self, ring):
        for i in range(5):
            ring.add("INFO", "t", f"line {i}")
        msgs = [e["message"] for e in ring.entries(limit=2)["entries"]]
        assert msgs == ["line 3", "line 4"]

    def test_redacts_on_the_way_in(self, ring):
        """The buffer is served over HTTP on a 0.0.0.0 bind, so a secret that
        reaches the deque is already exposed — scrubbing at render time is
        too late."""
        ring.add("INFO", "t", "token hf_abcdefghij0123456789")
        stored = ring.entries()["entries"][0]["message"]
        assert "hf_abcdefghij" not in stored
        assert "[redacted]" in stored

    def test_levels_are_ordered_low_to_high(self):
        assert LEVELS.index("INFO") < LEVELS.index("WARNING") < LEVELS.index("ERROR")


class TestRingHandler:
    def test_captures_records_once(self, ring):
        logger = logging.getLogger("gochidubb.test.logbuf")
        logger.propagate = False
        logger.setLevel(logging.INFO)
        handler = RingHandler(ring)
        logger.addHandler(handler)
        try:
            logger.warning("hello")
        finally:
            logger.removeHandler(handler)
        entries = ring.entries()["entries"]
        assert len(entries) == 1, "a duplicated line means stdio is teed twice"
        assert entries[0]["level"] == "WARNING"
        assert entries[0]["logger"] == "gochidubb.test.logbuf"
        assert entries[0]["message"] == "hello"

    def test_formats_lazy_args(self, ring):
        logger = logging.getLogger("gochidubb.test.logbuf2")
        logger.propagate = False
        logger.setLevel(logging.INFO)
        handler = RingHandler(ring)
        logger.addHandler(handler)
        try:
            logger.info("loaded %s in %ds", "model", 3)
        finally:
            logger.removeHandler(handler)
        assert ring.entries()["entries"][0]["message"] == "loaded model in 3s"

    def test_includes_the_traceback(self, ring):
        logger = logging.getLogger("gochidubb.test.logbuf3")
        logger.propagate = False
        handler = RingHandler(ring)
        logger.addHandler(handler)
        try:
            try:
                raise ValueError("boom")
            except ValueError:
                logger.exception("failed")
        finally:
            logger.removeHandler(handler)
        assert "ValueError: boom" in ring.entries()["entries"][0]["message"]


class _Sink:
    """Stand-in for a real stream, so tests never touch the console."""

    def __init__(self):
        self.written = []

    def write(self, s):
        self.written.append(s)
        return len(s)

    def flush(self):
        pass


class TestTeeStream:
    def test_passes_through_to_the_real_stream(self, ring):
        sink = _Sink()
        tee = TeeStream(sink, ring, "stdout")
        tee.write("hello\n")
        assert sink.written == ["hello\n"]

    def test_captures_third_party_print(self, ring):
        """The motivating case: pyannote prints its 'accept user conditions'
        banner with a bare print(), which no logging handler would ever see."""
        sink = _Sink()
        tee = TeeStream(sink, ring, "stdout")
        print("Could not download Model from pyannote/segmentation-3.0.", file=tee)
        msgs = [e["message"] for e in ring.entries()["entries"]]
        assert msgs == ["Could not download Model from pyannote/segmentation-3.0."]

    def test_holds_a_partial_line_until_its_newline(self, ring):
        sink = _Sink()
        tee = TeeStream(sink, ring, "stdout")
        tee.write("half a ")
        assert ring.entries()["entries"] == [], "no newline yet — nothing complete"
        tee.write("line\n")
        assert ring.entries()["entries"][0]["message"] == "half a line"

    def test_keeps_only_the_final_state_of_a_progress_bar(self, ring):
        """tqdm redraws with \\r; storing each redraw would flood the buffer."""
        sink = _Sink()
        tee = TeeStream(sink, ring, "stdout")
        tee.write("config.yaml:  10%\rconfig.yaml:  90%\rconfig.yaml: 100%\n")
        msgs = [e["message"] for e in ring.entries()["entries"]]
        assert msgs == ["config.yaml: 100%"]

    def test_drops_blank_lines(self, ring):
        sink = _Sink()
        tee = TeeStream(sink, ring, "stdout")
        tee.write("\n\n  \n")
        assert ring.entries()["entries"] == []

    def test_flushes_a_runaway_line_with_no_newline(self, ring):
        sink = _Sink()
        tee = TeeStream(sink, ring, "stdout")
        tee.write("x" * 9000)
        assert len(ring.entries()["entries"]) == 1

    def test_a_broken_stream_never_breaks_the_writer(self, ring):
        class Exploding:
            def write(self, s):
                raise OSError("pipe closed")

        tee = TeeStream(Exploding(), ring, "stdout")
        tee.write("still captured\n")   # must not raise
        assert ring.entries()["entries"][0]["message"] == "still captured"

    def test_isatty_survives_a_stream_without_one(self, ring):
        assert TeeStream(_Sink(), ring, "stdout").isatty() is False


class TestInstall:
    def test_is_idempotent_and_restores_stdio(self):
        from app import logbuf

        # Importing server (other test modules do) already installed it, and
        # pytest swapped sys.stdout afterwards — start from a known state.
        logbuf.uninstall()
        before_out, before_err = sys.stdout, sys.stderr
        root_handlers = len(logging.getLogger().handlers)
        logbuf.install()
        try:
            logbuf.install()   # second call must be a no-op
            added = [h for h in logging.getLogger().handlers
                     if isinstance(h, RingHandler)]
            assert len(added) == 1, "a second handler would duplicate every line"
            assert sys.stdout is not before_out, "stdout should be teed"
        finally:
            logbuf.uninstall()
        assert sys.stdout is before_out
        assert sys.stderr is before_err
        assert len(logging.getLogger().handlers) == root_handlers

    def test_console_handler_bypasses_the_tee(self):
        """The double-capture trap: basicConfig's StreamHandler writes to
        sys.stderr, so teeing stderr naively records every log line twice —
        once as a record, once as the handler's own output."""
        from app import logbuf

        logbuf.uninstall()
        root = logging.getLogger()
        sink = _Sink()
        console = logging.StreamHandler(sink)
        root.addHandler(console)
        logbuf.install()
        try:
            logging.getLogger("gochidubb.test.dup").warning("only once please")
            msgs = [e["message"] for e in logbuf.ring.entries(limit=50)["entries"]
                    if "only once please" in e["message"]]
            assert len(msgs) == 1
        finally:
            logbuf.uninstall()
            root.removeHandler(console)
