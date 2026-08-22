"""Tests for the stage-based pipeline: checkpoints, partial retries, metrics.

These exercise server.py's stage machinery directly rather than through
HTTP, because the interesting behaviour is in how a stage decides what to
reuse from a previous run and what to redo.
"""
import asyncio
import sys
import types
from pathlib import Path

import pytest

# server.py transitively imports the heavy optional ML stack. Stub the
# modules that may not be installed in a test environment so the import
# works on a bare checkout.
for _name in ("voxcpm", "whisperx", "faster_whisper", "pyannote", "edge_tts", "demucs"):
    sys.modules.setdefault(_name, types.ModuleType(_name))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
server = pytest.importorskip("server")


@pytest.fixture
def job_env(tmp_path, monkeypatch):
    """An isolated job directory with server globals pointed at it."""
    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(server, "save_job", lambda job: None)
    job_id = "testjob"
    work = tmp_path / job_id
    work.mkdir()
    job = {"id": job_id, "status": "queued"}
    monkeypatch.setitem(server.jobs, job_id, job)
    return types.SimpleNamespace(job_id=job_id, job=job, work=work)


def _update_noop(**kwargs):
    pass


# ── Registry invariants ──────────────────────────────────────────────

class TestStageRegistry:
    def test_every_stage_has_a_handler(self):
        assert set(server.STAGE_ORDER) == set(server.STAGE_HANDLERS)

    def test_checkpoint_names_are_unique(self):
        names = [s["checkpoint"] for s in server.PIPELINE_STAGES]
        assert len(names) == len(set(names))

    def test_legacy_checkpoint_names_preserved(self):
        """Older endpoints read these filenames directly — don't rename them."""
        by_id = {s["id"]: s["checkpoint"] for s in server.PIPELINE_STAGES}
        assert by_id["diarize"] == "transcription_done"
        assert by_id["translate"] == "translation_done"
        assert by_id["tts"] == "tts_done"

    def test_checkpoint_order_desc_is_newest_first(self):
        assert server.CHECKPOINT_ORDER_DESC[0] == "merge_done"
        assert server.CHECKPOINT_ORDER_DESC[-1] == "download_done"

    def test_stage_index_unknown_is_negative(self):
        assert server._stage_index("nope") == -1
        assert server._stage_index("translate") == 4

    @pytest.mark.parametrize("checkpoint,expected", [
        ("transcribe_done", "diarize"),
        ("transcription_done", "translate"),
        ("translation_done", "tts"),
        ("merge_done", ""),
        ("some_legacy_name", "tts"),
    ])
    def test_stage_after_checkpoint(self, checkpoint, expected):
        assert server._stage_after_checkpoint(checkpoint) == expected


# ── Context serialization ────────────────────────────────────────────

class TestContextSnapshot:
    def test_drops_private_keys(self):
        out = server._ctx_for_checkpoint({"model": "x", "_is_retry": True})
        assert out == {"model": "x"}

    def test_drops_unserializable_values(self):
        out = server._ctx_for_checkpoint({"ok": 1, "bad": object()})
        assert out == {"ok": 1}

    def test_serialize_segments_assigns_stable_idx(self):
        segs = server._serialize_segments(
            [{"start": 0, "end": 1, "text": "a"}, {"start": 1, "end": 2, "text": "b"}]
        )
        assert [s["idx"] for s in segs] == [0, 1]
        assert segs[0]["speaker"] == "SPEAKER_00"

    def test_serialize_segments_preserves_existing_idx(self):
        segs = server._serialize_segments([{"idx": 7, "start": 0, "end": 1, "text": "a"}])
        assert segs[0]["idx"] == 7

    def test_serialize_segments_omits_absent_optionals(self):
        segs = server._serialize_segments([{"start": 0, "end": 1, "text": "a"}])
        assert "audio_path" not in segs[0]
        assert "translated_text" not in segs[0]


# ── Checkpoint lookup ────────────────────────────────────────────────

class TestStageInputState:
    def test_first_stage_has_no_input(self, job_env):
        assert server._stage_input_state(job_env.job_id, "download") is None

    def test_walks_back_to_the_nearest_checkpoint(self, job_env):
        """A job with only the legacy transcription checkpoint is still
        retryable from translate AND from tts."""
        server._save_checkpoint(job_env.job_id, job_env.work,
                                stage="transcription_done", data={"segments": []})
        for stage in ("translate", "tts", "merge"):
            state = server._stage_input_state(job_env.job_id, stage)
            assert state is not None, stage
            assert state["stage"] == "transcription_done"

    def test_prefers_the_immediately_preceding_checkpoint(self, job_env):
        server._save_checkpoint(job_env.job_id, job_env.work,
                                stage="transcription_done", data={"segments": []})
        server._save_checkpoint(job_env.job_id, job_env.work,
                                stage="translation_done", data={"segments": []})
        state = server._stage_input_state(job_env.job_id, "tts")
        assert state["stage"] == "translation_done"

    def test_returns_none_when_nothing_was_saved(self, job_env):
        assert server._stage_input_state(job_env.job_id, "tts") is None


# ── Untranslated detection ───────────────────────────────────────────

class TestUntranslatedDetection:
    @pytest.mark.parametrize("seg,expected", [
        ({"text": "hello", "translated_text": "привет"}, False),
        ({"text": "hello", "translated_text": ""}, True),
        ({"text": "hello"}, True),
        ({"text": "hello", "translated_text": "hello"}, True),
        ({"text": "hello", "translated_text": "  hello  "}, True),
    ])
    def test_is_untranslated(self, seg, expected):
        assert server._is_untranslated(seg) is expected


# ── Partial translation retry ────────────────────────────────────────

class TestPartialTranslateRetry:
    """`translate_failed_only` keeps good work and redoes only failures."""

    def _run(self, job_env, monkeypatch, ctx, translator):
        monkeypatch.setattr(server, "translate_segments", translator)
        async def _noop_unload(model):
            return None
        monkeypatch.setattr(server, "unload_ollama_model", _noop_unload)
        monkeypatch.setattr(server, "write_srt", lambda segs, p: Path(p).write_text("srt"))
        perf = {}
        asyncio.run(server._stage_translate(
            job_env.job, job_env.work, ctx, _update_noop, perf))
        return perf

    def test_only_failed_segments_are_resubmitted(self, job_env, monkeypatch):
        # Previous run: segment 0 translated, segment 1 came back empty.
        server._save_checkpoint(job_env.job_id, job_env.work, stage="translation_done", data={
            "segments": [
                {"idx": 0, "start": 0, "end": 1, "text": "one", "translated_text": "один"},
                {"idx": 1, "start": 1, "end": 2, "text": "two", "translated_text": ""},
            ]})
        seen = []

        async def translator(segs, tgt, model, context_hint=None, progress_callback=None, **kw):
            seen.append([s["idx"] for s in segs])
            return [dict(s, translated_text=f"{model}:{s['text']}") for s in segs]

        ctx = {
            "target_lang": "ru", "model": "new-model", "translate_failed_only": True,
            "segments": [
                {"idx": 0, "start": 0, "end": 1, "text": "one", "speaker": "S"},
                {"idx": 1, "start": 1, "end": 2, "text": "two", "speaker": "S"},
            ],
        }
        perf = self._run(job_env, monkeypatch, ctx, translator)

        assert seen == [[1]], "only the failed segment should be re-translated"
        assert perf["reused"] == 1 and perf["translated"] == 1
        by_idx = {s["idx"]: s for s in ctx["segments"]}
        assert by_idx[0]["translated_text"] == "один"          # untouched
        assert by_idx[1]["translated_text"] == "new-model:two"  # redone

    def test_prior_translation_discarded_when_source_text_changed(self, job_env, monkeypatch):
        """If transcription was re-run, segment N is a different sentence —
        reusing its old translation would silently mistranslate the line."""
        server._save_checkpoint(job_env.job_id, job_env.work, stage="translation_done", data={
            "segments": [
                {"idx": 0, "start": 0, "end": 1, "text": "OLD TEXT",
                 "translated_text": "старый перевод"},
            ]})
        seen = []

        async def translator(segs, tgt, model, context_hint=None, progress_callback=None, **kw):
            seen.append([s["text"] for s in segs])
            return [dict(s, translated_text=f"new:{s['text']}") for s in segs]

        ctx = {
            "target_lang": "ru", "model": "m", "translate_failed_only": True,
            "segments": [{"idx": 0, "start": 0, "end": 1, "text": "NEW TEXT", "speaker": "S"}],
        }
        perf = self._run(job_env, monkeypatch, ctx, translator)

        assert seen == [["NEW TEXT"]]
        assert perf["reused"] == 0
        assert ctx["segments"][0]["translated_text"] == "new:NEW TEXT"

    def test_nothing_to_do_short_circuits(self, job_env, monkeypatch):
        server._save_checkpoint(job_env.job_id, job_env.work, stage="translation_done", data={
            "segments": [{"idx": 0, "start": 0, "end": 1, "text": "one",
                          "translated_text": "один"}]})
        called = []

        async def translator(segs, tgt, model, context_hint=None, progress_callback=None, **kw):
            called.append(1)
            return segs

        ctx = {
            "target_lang": "ru", "model": "m", "translate_failed_only": True,
            "segments": [{"idx": 0, "start": 0, "end": 1, "text": "one", "speaker": "S"}],
        }
        perf = self._run(job_env, monkeypatch, ctx, translator)
        assert called == [] and perf["translated"] == 0
        assert (job_env.work / "subtitles.srt").exists()

    def test_total_failure_raises_with_actionable_message(self, job_env, monkeypatch):
        async def translator(segs, tgt, model, context_hint=None, progress_callback=None, **kw):
            return [dict(s, translated_text=s["text"]) for s in segs]

        ctx = {"target_lang": "ru", "model": "broken",
               "segments": [{"idx": 0, "start": 0, "end": 1, "text": "one", "speaker": "S"}]}
        with pytest.raises(RuntimeError, match="Translation failed for 1 of 1"):
            self._run(job_env, monkeypatch, ctx, translator)

    def test_mostly_untranslated_is_a_failure_not_a_pass(self, job_env, monkeypatch):
        """A run that translated 16 of 182 lines was recorded status=ok and
        flowed into TTS, producing a "finished" dub that was 91% English."""
        async def translator(segs, tgt, model, context_hint=None, progress_callback=None, **kw):
            out = []
            for i, s in enumerate(segs):
                # only the first line comes back translated
                out.append(dict(s, translated_text="перевод" if i == 0 else s["text"]))
            return out

        ctx = {"target_lang": "ru", "model": "slow",
               "segments": [{"idx": i, "start": i, "end": i + 1,
                             "text": f"line {i}", "speaker": "S"} for i in range(10)]}
        with pytest.raises(RuntimeError, match="Translation failed for 9 of 10"):
            self._run(job_env, monkeypatch, ctx, translator)

    def test_a_mostly_successful_run_still_passes(self, job_env, monkeypatch):
        async def translator(segs, tgt, model, context_hint=None, progress_callback=None, **kw):
            out = []
            for i, s in enumerate(segs):
                out.append(dict(s, translated_text=s["text"] if i == 0 else "перевод"))
            return out

        ctx = {"target_lang": "ru", "model": "ok",
               "segments": [{"idx": i, "start": i, "end": i + 1,
                             "text": f"line {i}", "speaker": "S"} for i in range(10)]}
        self._run(job_env, monkeypatch, ctx, translator)   # must not raise


# ── Partial TTS retry ────────────────────────────────────────────────

class FakeTTS:
    sample_rate = 24000

    def __init__(self):
        self.rendered = []

    def synthesize_segments(self, segs, out_dir, **kwargs):
        self.rendered.append([s["idx"] for s in segs])
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for s in segs:
            p = Path(out_dir) / f"seg{s['idx']}.wav"
            p.write_bytes(b"wav")
            s["audio_path"] = str(p)
        return segs


class TestPartialTTSRetry:
    def _prepare(self, job_env, monkeypatch):
        tts = FakeTTS()
        monkeypatch.setattr(server, "get_tts_engine", lambda: tts)
        monkeypatch.setattr(server, "VoxCPMSynthesizer", FakeTTS)
        return tts

    def test_recovers_rendered_audio_from_the_tts_checkpoint(self, job_env, monkeypatch):
        """Retrying `tts` starts from the translation checkpoint, which has
        no audio paths — they must be recovered so 'only missing' works."""
        tts = self._prepare(job_env, monkeypatch)
        existing = job_env.work / "old0.wav"
        existing.write_bytes(b"wav")
        server._save_checkpoint(job_env.job_id, job_env.work, stage="tts_done", data={
            "segments": [
                {"idx": 0, "start": 0, "end": 1, "text": "one", "translated_text": "один",
                 "audio_path": str(existing)},
                {"idx": 1, "start": 1, "end": 2, "text": "two", "translated_text": "два",
                 "audio_path": ""},
            ]})
        ctx = {
            "tts_keep_existing": True, "effective_src": "en", "target_lang": "ru",
            "segments": [
                {"idx": 0, "start": 0, "end": 1, "text": "one", "translated_text": "один", "speaker": "S"},
                {"idx": 1, "start": 1, "end": 2, "text": "two", "translated_text": "два", "speaker": "S"},
            ],
        }
        asyncio.run(server._stage_tts(job_env.job, job_env.work, ctx, _update_noop, {}))

        assert tts.rendered == [[1]], "only the unrendered segment should synthesize"
        by_idx = {s["idx"]: s for s in ctx["segments"]}
        assert by_idx[0]["audio_path"] == str(existing)

    def test_edited_translation_forces_resynthesis(self, job_env, monkeypatch):
        """A line whose translation changed must not keep its old audio."""
        tts = self._prepare(job_env, monkeypatch)
        stale = job_env.work / "stale.wav"
        stale.write_bytes(b"wav")
        server._save_checkpoint(job_env.job_id, job_env.work, stage="tts_done", data={
            "segments": [{"idx": 0, "start": 0, "end": 1, "text": "one",
                          "translated_text": "СТАРЫЙ", "audio_path": str(stale)}]})
        ctx = {
            "tts_keep_existing": True, "effective_src": "en", "target_lang": "ru",
            "segments": [{"idx": 0, "start": 0, "end": 1, "text": "one",
                          "translated_text": "НОВЫЙ", "speaker": "S"}],
        }
        asyncio.run(server._stage_tts(job_env.job, job_env.work, ctx, _update_noop, {}))
        assert tts.rendered == [[0]]
        assert ctx["segments"][0]["audio_path"] != str(stale)

    def test_retry_rolls_a_fresh_voice_seed(self, job_env, monkeypatch):
        """Identical inputs give VoxCPM identical output, so a retry that
        kept the seed would look like it did nothing."""
        self._prepare(job_env, monkeypatch)
        ctx = {
            "_is_retry": True, "voice_seed": 1234, "effective_src": "en", "target_lang": "ru",
            "source_speaker_refs": {}, "speaker_refs": {},
            "segments": [{"idx": 0, "start": 0, "end": 1, "text": "one",
                          "translated_text": "один", "speaker": "S"}],
        }
        asyncio.run(server._stage_tts(job_env.job, job_env.work, ctx, _update_noop, {}))
        assert ctx["voice_seed"] != 1234


# ── Stage report ─────────────────────────────────────────────────────

class TestStageReport:
    def test_pending_when_nothing_ran(self, job_env):
        rep = server.build_stage_report(job_env.job_id)
        assert [s["state"] for s in rep["stages"]] == ["pending"] * len(server.STAGE_ORDER)
        assert rep["total_duration_sec"] == 0

    def test_marks_downstream_checkpoints_stale_after_an_upstream_rerun(self, job_env):
        """Re-running translate alone leaves the old tts output on disk; it
        must not be presented as current."""
        import time
        server._save_checkpoint(job_env.job_id, job_env.work,
                                stage="translation_done", data={"segments": []})
        server._save_checkpoint(job_env.job_id, job_env.work,
                                stage="tts_done", data={"segments": []})
        time.sleep(0.01)
        # Upstream stage re-run after the downstream one
        server._save_checkpoint(job_env.job_id, job_env.work,
                                stage="transcription_done", data={"segments": []})

        states = {s["id"]: s["state"] for s in server.build_stage_report(job_env.job_id)["stages"]}
        assert states["diarize"] == "done"
        assert states["translate"] == "stale"
        assert states["tts"] == "stale"

    def test_failed_stage_is_reported(self, job_env):
        job_env.job.update(status="error", failed_stage="translate")
        states = {s["id"]: s["state"] for s in server.build_stage_report(job_env.job_id)["stages"]}
        assert states["translate"] == "failed"

    def test_running_stage_is_reported_and_blocks_retry(self, job_env):
        job_env.job.update(status="transcribing", stage_id="transcribe")
        rep = server.build_stage_report(job_env.job_id)
        states = {s["id"]: s for s in rep["stages"]}
        assert rep["busy"] is True
        assert states["transcribe"]["state"] == "running"
        assert all(not s["can_retry"] for s in rep["stages"])

    def test_timings_and_resources_come_from_metrics(self, job_env):
        from pipeline.metrics import record_stage
        record_stage(job_env.work, job_env.job_id, "transcribe", "ok", 12.5, 1000.0,
                     resources={"cpu_proc_pct_peak": 91.0}, detail={"segments": 42})
        rep = server.build_stage_report(job_env.job_id)
        st = next(s for s in rep["stages"] if s["id"] == "transcribe")
        assert st["duration_sec"] == 12.5
        assert st["attempt"] == 1
        assert st["resources"]["cpu_proc_pct_peak"] == 91.0
        assert st["detail"]["segments"] == 42
        assert rep["total_duration_sec"] == 12.5

    def test_degraded_surfaces_a_stage_that_succeeded_the_wrong_way(self, job_env):
        """Diarization falling back to a weaker model finishes "ok". Without a
        degraded flag that is indistinguishable from a clean run, which is how
        the whole problem stayed invisible."""
        from pipeline.metrics import record_stage
        from pipeline.notices import notice

        server._save_checkpoint(job_env.job_id, job_env.work,
                                stage="transcription_done", data={"segments": []})
        record_stage(job_env.work, job_env.job_id, "diarize", "ok", 9.0, 1000.0,
                     detail={"speaker_turns": 0,
                             "notices": [notice("pyannote.fallback_model", "warn",
                                                "Used the fallback model",
                                                url="https://hf.co/pyannote/segmentation-3.0")]})
        rep = server.build_stage_report(job_env.job_id)
        st = next(s for s in rep["stages"] if s["id"] == "diarize")
        assert st["state"] == "done", "the transcript is still usable"
        assert st["degraded"] is True
        assert [n["code"] for n in st["notices"]] == ["pyannote.fallback_model"]
        assert [n["code"] for n in rep["notices"]] == ["pyannote.fallback_model"]

    def test_no_notices_means_not_degraded(self, job_env):
        from pipeline.metrics import record_stage
        record_stage(job_env.work, job_env.job_id, "diarize", "ok", 9.0, 1000.0,
                     detail={"speaker_turns": 6})
        st = next(s for s in server.build_stage_report(job_env.job_id)["stages"]
                  if s["id"] == "diarize")
        assert st["degraded"] is False
        assert st["notices"] == []

    def test_pending_stage_with_notices_is_not_degraded(self, job_env):
        """`degraded` describes a completed stage; a stage that failed already
        shows its error, and one that never ran has nothing to qualify."""
        from pipeline.metrics import record_stage
        from pipeline.notices import notice
        record_stage(job_env.work, job_env.job_id, "diarize", "error", 1.0, 1000.0,
                     detail={"notices": [notice("pyannote.load_failed", "error", "x")]},
                     error="RuntimeError: nope")
        st = next(s for s in server.build_stage_report(job_env.job_id)["stages"]
                  if s["id"] == "diarize")
        assert st["state"] == "failed"
        assert st["degraded"] is False
        assert st["notices"], "the notice still travels with the stage"

    def test_report_notices_are_deduped_across_stages(self, job_env):
        from pipeline.metrics import record_stage
        from pipeline.notices import notice
        dup = [notice("tts_qa.device_unavailable", "warn", "QA off")]
        for sid in ("diarize", "tts"):
            record_stage(job_env.work, job_env.job_id, sid, "ok", 1.0, 1000.0,
                         detail={"notices": dup})
        rep = server.build_stage_report(job_env.job_id)
        assert [n["code"] for n in rep["notices"]] == ["tts_qa.device_unavailable"]

    def test_artifacts_found_by_well_known_filename(self, job_env):
        """Jobs created before stage checkpoints existed still show output."""
        server._save_checkpoint(job_env.job_id, job_env.work,
                                stage="translation_done", data={"segments": []})
        (job_env.work / "subtitles.srt").write_text("1\n")
        rep = server.build_stage_report(job_env.job_id)
        tr = next(s for s in rep["stages"] if s["id"] == "translate")
        assert [a["name"] for a in tr["artifacts"]] == ["subtitles.srt"]
        assert tr["artifacts"][0]["url"].endswith("/subtitles.srt")


# ── Retry override safety ────────────────────────────────────────────

class TestRetryOverrides:
    def test_artifact_paths_are_not_overridable(self):
        """A client must not be able to repoint the pipeline at arbitrary
        files on disk via the overrides payload."""
        for key in ("video_path", "audio_16k", "speaker_refs", "segments",
                    "dubbed_wav", "output_mp4", "source"):
            assert key not in server._RETRY_OVERRIDE_KEYS

    def test_documented_options_are_all_overridable(self):
        for stage, opts in server.STAGE_RETRY_OPTIONS.items():
            for opt in opts:
                assert opt["key"] in server._RETRY_OVERRIDE_KEYS, (stage, opt["key"])

    def test_every_stage_has_an_options_entry(self):
        assert set(server.STAGE_RETRY_OPTIONS) == set(server.STAGE_ORDER)

    def test_numeric_options_declare_their_bounds(self):
        """The UI renders these generically, so min/max/step have to come
        from the declaration — otherwise a number input offers no bounds at
        all and every out-of-range value becomes a round trip to a 400."""
        for stage, opts in server.STAGE_RETRY_OPTIONS.items():
            for opt in opts:
                if opt.get("type") == "number":
                    assert {"min", "max", "step"} <= set(opt), (stage, opt["key"])

    def test_numeric_retry_keys_are_the_ones_that_need_coercing(self):
        """_RETRY_NUMERIC_KEYS drives coercion in retry_stage. A number
        declared in the options table but missing here would reach VoxCPM as
        whatever string the client posted."""
        declared = {opt["key"] for opts in server.STAGE_RETRY_OPTIONS.values()
                    for opt in opts if opt.get("type") == "number"}
        assert declared == set(server._RETRY_NUMERIC_KEYS)

    def test_per_job_voxcpm_overrides_are_retryable(self):
        """Re-running just the TTS stage with a different guidance is the
        cheapest way to use this feature — no re-transcribe, no re-translate."""
        for key in ("voxcpm_cfg", "voxcpm_steps"):
            assert key in server._RETRY_OVERRIDE_KEYS


# ── Event-loop responsiveness ────────────────────────────────────────

class TestNonBlockingStages:
    """The UI polls /api/jobs and /api/dub/{id}/stages every few seconds. If a
    stage runs its blocking work on the event loop, those polls queue behind
    it for minutes — the stage panel shows "Loading stages…" for the whole run
    and a new job doesn't appear as running until the current stage finishes.
    """

    def test_offload_runs_the_call_off_the_event_loop(self):
        import threading

        loop_thread = threading.get_ident()
        seen = {}

        def blocking(x):
            seen["thread"] = threading.get_ident()
            return x * 2

        async def go():
            return await server._blocking(blocking, 21)

        assert asyncio.run(go()) == 42
        assert seen["thread"] != loop_thread, "must not run on the loop thread"

    def test_offload_forwards_keyword_arguments(self):
        def blocking(a, b=0, c=0):
            return a + b + c

        async def go():
            return await server._blocking(blocking, 1, b=2, c=3)

        assert asyncio.run(go()) == 6

    def test_offload_propagates_exceptions(self):
        def blocking():
            raise RuntimeError("boom")

        async def go():
            return await server._blocking(blocking)

        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(go())

    def test_loop_stays_free_while_a_stage_blocks(self):
        """The actual regression: something else must get scheduled meanwhile."""
        import time as _time

        ticks = []

        async def ticker():
            for _ in range(20):
                ticks.append(1)
                await asyncio.sleep(0.01)

        async def go():
            t = asyncio.create_task(ticker())
            await server._blocking(_time.sleep, 0.15)
            t.cancel()

        asyncio.run(go())
        assert len(ticks) > 1, "a blocking stage starved the event loop"

    def test_every_blocking_pipeline_call_is_offloaded(self):
        """Guard against a new stage quietly reintroducing the freeze."""
        import inspect
        import re

        BLOCKING = ("download_video", "extract_audio", "extract_audio_hq",
                    "separate_background", "apply_vad_filter", "transcribe",
                    "diarize_speakers", "extract_speaker_audio",
                    "extract_fallback_reference", "assemble_dubbed_audio",
                    "merge_audio_video")
        offenders = []
        for sid, handler in server.STAGE_HANDLERS.items():
            src = inspect.getsource(handler)
            for name in BLOCKING:
                # A bare `name(` not preceded by `_blocking,` on the line above.
                for m in re.finditer(rf"(?<![\w.]){re.escape(name)}\s*\(", src):
                    line_start = src.rfind("\n", 0, m.start()) + 1
                    prefix = src[max(0, line_start - 80):m.start()]
                    if "_blocking" not in prefix:
                        offenders.append(f"{sid}: {name}")
        assert not offenders, (
            "these run on the event loop and will freeze the UI: "
            + ", ".join(sorted(set(offenders)))
        )


# ── Queue payload / run_pipeline agreement ───────────────────────────

class TestEnqueuePayloads:
    """Everything a route enqueues is splatted into run_pipeline as keyword
    arguments (`await run_pipeline(job_id, **pipeline_args)`), so a key the
    signature doesn't have is a TypeError at DEQUEUE — after the job has been
    accepted, persisted and shown as queued. That is how `lip_sync` in the
    multi-language route killed every job it created: the flag belongs on the
    job dict (the post-success hook reads it there), not in the pipeline args.
    """

    @staticmethod
    def _pipeline_params():
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(server))
        fn = next(n for n in tree.body
                  if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_pipeline")
        assert fn.args.kwarg is None, (
            "run_pipeline grew **kwargs — this check silently stops working")
        return ({a.arg for a in fn.args.args} |
                {a.arg for a in fn.args.kwonlyargs})

    def test_every_enqueue_site_binds(self):
        import ast
        import inspect
        params = self._pipeline_params()
        tree = ast.parse(inspect.getsource(server))
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name not in ("enqueue_job", "_enqueue_or_defer"):
                continue
            if len(node.args) < 2 or not isinstance(node.args[1], ast.Dict):
                continue
            for k in node.args[1].keys:
                # __stage_retry__ is the queue's other payload shape and does
                # not reach run_pipeline at all.
                if (isinstance(k, ast.Constant) and k.value not in params
                        and not str(k.value).startswith("__")):
                    offenders.append(f"line {node.lineno}: {k.value}")
        assert not offenders, (
            "run_pipeline cannot accept these enqueued keys: "
            + ", ".join(offenders))

    def test_every_scheduled_stash_binds(self):
        """_pending_args is replayed through the same call by the scheduler,
        so a deferred job must not fail hours later for this reason."""
        import ast
        import inspect
        params = self._pipeline_params()
        tree = ast.parse(inspect.getsource(server))
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for k, v in zip(node.keys, node.values):
                if not (isinstance(k, ast.Constant) and k.value == "_pending_args"):
                    continue
                d = v.body if isinstance(v, ast.IfExp) else v
                if not isinstance(d, ast.Dict):
                    continue
                for kk in d.keys:
                    if isinstance(kk, ast.Constant) and kk.value not in params:
                        offenders.append(f"line {k.lineno}: {kk.value}")
        assert not offenders, (
            "these stashed keys would fail when the scheduler replays them: "
            + ", ".join(offenders))


class TestPerJobVoxcpmOverrides:
    """The route accepts a number, the pipeline puts it in ctx, and the TTS
    stage hands it to the engine. Each hop is cheap to break silently.
    """

    def test_run_pipeline_puts_the_override_in_ctx(self, monkeypatch):
        """ctx is what every stage reads, and what gets checkpointed — so a
        retry hours later replays the same value."""
        seen = {}

        async def fake_stages(job_id, ctx, **kw):
            seen.update(ctx)

        monkeypatch.setattr(server, "run_pipeline_stages", fake_stages)
        monkeypatch.setitem(server.jobs, "j1", {"id": "j1"})
        asyncio.run(server.run_pipeline(
            "j1", source="x", source_lang="en", target_lang="ru",
            model="m", keep_bg=False, whisper_model="tiny",
            voxcpm_cfg=2.4, voxcpm_steps=18))
        assert seen["voxcpm_cfg"] == 2.4
        assert seen["voxcpm_steps"] == 18

    def test_default_is_the_inherit_sentinel(self):
        """An old queued job, or any caller that doesn't set them, must keep
        following the global setting rather than pinning a value."""
        import inspect
        sig = inspect.signature(server.run_pipeline)
        assert sig.parameters["voxcpm_cfg"].default == 0.0
        assert sig.parameters["voxcpm_steps"].default == 0

    def test_the_tts_stage_forwards_them_to_the_engine(self):
        """_stage_tts is the single call site for a normal run; if it stops
        passing the overrides the whole feature is inert with no error."""
        import inspect
        src = inspect.getsource(server._stage_tts)
        assert 'cfg_override=ctx.get("voxcpm_cfg")' in src
        assert 'steps_override=ctx.get("voxcpm_steps")' in src

    def test_the_rerun_path_reads_them_off_the_job(self):
        """Redub, retry_tts and per-segment regen all go through here."""
        import inspect
        src = inspect.getsource(server._run_tts_and_merge_stage)
        assert 'cfg_override=job.get("voxcpm_cfg")' in src
        assert 'steps_override=job.get("voxcpm_steps")' in src
