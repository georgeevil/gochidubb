"""Tests for tools/audit_job.py — the pipeline information-loss auditor.

The auditor exists because segment counts alone cannot tell "restructured"
from "lost": the pipeline deliberately merges sentence fragments and splits
over-long segments, so 190 → 152 is healthy while 152 → 150 is two lines of
dialogue gone. These tests pin that distinction.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
audit = pytest.importorskip("audit_job")


def _seg(idx, start, end, text, **extra):
    s = {"idx": idx, "start": start, "end": end, "text": text,
         "speaker": "SPEAKER_00"}
    s.update(extra)
    return s


def _job(tmp_path, **checkpoints):
    """Build an output directory from {stage_checkpoint_name: segments}."""
    work = tmp_path / "job1"
    work.mkdir(exist_ok=True)
    for cp, segs in checkpoints.items():
        (work / f"checkpoint_{cp}.json").write_text(
            json.dumps({"segments": segs}), encoding="utf-8")
    return work


def _codes(report, severity="loss"):
    return [f["summary"] for f in report["findings"] if f["severity"] == severity]


def _has(report, needle, severity="loss"):
    return any(needle in s for s in _codes(report, severity))


HEALTHY = [_seg(0, 0.0, 2.0, "hello there friend"),
           _seg(1, 2.0, 4.0, "this is a second line"),
           _seg(2, 4.0, 6.0, "and a third one here")]


def _synthesized(work, segs):
    """Give each segment a real wav on disk, as a completed TTS stage would."""
    seg_dir = work / "tts_segments"
    seg_dir.mkdir(exist_ok=True)
    out = []
    for s in segs:
        wav = seg_dir / f"seg_{s['idx']:04d}.wav"
        wav.write_bytes(b"\x00" * 4096)
        out.append(dict(s, audio_path=str(wav)))
    return out


class TestCleanJob:
    def test_no_findings_when_nothing_changes(self, tmp_path):
        work = tmp_path / "job1"
        work.mkdir()
        segs = [dict(s, translated_text="перевод") for s in HEALTHY]
        segs = _synthesized(work, segs)
        for cp in ("transcribe_done", "transcription_done"):
            (work / f"checkpoint_{cp}.json").write_text(
                json.dumps({"segments": HEALTHY}), encoding="utf-8")
        for cp in ("translation_done", "tts_done", "assemble_done"):
            (work / f"checkpoint_{cp}.json").write_text(
                json.dumps({"segments": segs}), encoding="utf-8")
        (work / "tts_placements.json").write_text(
            json.dumps([{"idx": s["idx"]} for s in segs]), encoding="utf-8")
        r = audit.audit_job(work)
        assert r["ok"], _codes(r)

    def test_missing_directory_is_reported(self, tmp_path):
        r = audit.audit_job(tmp_path / "nope")
        assert not r["ok"]

    def test_job_with_no_checkpoints_is_not_a_loss(self, tmp_path):
        """A job that died before its first checkpoint has nothing to lose —
        flagging it would make --all cry wolf on every dead job."""
        work = tmp_path / "empty"
        work.mkdir()
        r = audit.audit_job(work)
        assert r["ok"]
        assert r["counts"]["info"] == 1


class TestDuplicateIndex:
    """The bug that motivated the tool (job 947ca81d)."""

    def test_duplicate_idx_is_flagged_with_both_texts(self, tmp_path):
        dup = HEALTHY + [_seg(1, 6.0, 8.0, "the line that gets overwritten")]
        work = _job(tmp_path, transcribe_done=HEALTHY, transcription_done=dup)
        r = audit.audit_job(work)
        assert _has(r, "duplicate idx")
        finding = next(f for f in r["findings"] if "duplicate idx" in f["summary"])
        texts = finding["items"][0]["texts"]
        assert "this is a second line" in texts
        assert "the line that gets overwritten" in texts

    def test_collapse_after_duplicates_explains_itself(self, tmp_path):
        """152 -> 150 with no missing index is the signature of an idx-keyed
        dict collapsing duplicates; the report should say so rather than
        leaving the reader hunting for a missing index."""
        dup = HEALTHY + [_seg(1, 6.0, 8.0, "overwritten")]
        work = _job(tmp_path, transcribe_done=HEALTHY, transcription_done=dup,
                    translation_done=HEALTHY)
        r = audit.audit_job(work)
        finding = next(f for f in r["findings"]
                       if "segments (1 lost)" in f["summary"])
        assert "duplicate idx" in finding["detail"]


class TestReshapingIsNotLoss:
    def test_merging_segments_is_not_reported_as_loss(self, tmp_path):
        """transcribe -> diarize legitimately merges fragments."""
        merged = [_seg(0, 0.0, 4.0, "hello there friend this is a second line"),
                  _seg(1, 4.0, 6.0, "and a third one here")]
        work = _job(tmp_path, transcribe_done=HEALTHY, transcription_done=merged,
                    translation_done=[dict(s, translated_text="x") for s in merged])
        r = audit.audit_job(work)
        assert r["ok"], _codes(r)

    def test_splitting_segments_is_not_reported_as_loss(self, tmp_path):
        split = [_seg(0, 0.0, 1.0, "hello there"), _seg(1, 1.0, 2.0, "friend"),
                 _seg(2, 2.0, 4.0, "this is a second line"),
                 _seg(3, 4.0, 6.0, "and a third one here")]
        work = _job(tmp_path, transcribe_done=HEALTHY, transcription_done=split,
                    translation_done=[dict(s, translated_text="x") for s in split])
        r = audit.audit_job(work)
        assert r["ok"], _codes(r)

    def test_dropping_a_segment_during_reshaping_is_caught_by_words(self, tmp_path):
        """Counts alone cannot catch this — the whole reason for word coverage."""
        lossy = [_seg(0, 0.0, 4.0, "hello there friend this is a second line")]
        work = _job(tmp_path, transcribe_done=HEALTHY, transcription_done=lossy,
                    translation_done=[dict(s, translated_text="x") for s in lossy])
        r = audit.audit_job(work)
        assert _has(r, "words")
        finding = next(f for f in r["findings"] if "words" in f["summary"])
        assert any("third one" in i["text"] for i in finding["items"])


class TestNonReshapingBoundaries:
    def test_segment_lost_between_translate_and_tts(self, tmp_path):
        tr = [dict(s, translated_text="x") for s in HEALTHY]
        work = _job(tmp_path, transcription_done=HEALTHY,
                    translation_done=tr, tts_done=tr[:2])
        r = audit.audit_job(work)
        assert _has(r, "3 → 2 segments")
        finding = next(f for f in r["findings"] if "3 → 2" in f["summary"])
        assert finding["items"][0]["text"] == "and a third one here"


class TestTranslationChecks:
    def test_untranslated_segments_are_listed(self, tmp_path):
        segs = [dict(HEALTHY[0], translated_text="перевод"),
                dict(HEALTHY[1], translated_text=""),
                dict(HEALTHY[2], translated_text=HEALTHY[2]["text"])]
        work = _job(tmp_path, translation_done=segs)
        r = audit.audit_job(work)
        assert _has(r, "no translation")
        finding = next(f for f in r["findings"] if "no translation" in f["summary"])
        assert {i["idx"] for i in finding["items"]} == {1, 2}


class TestTtsChecks:
    def test_segment_never_synthesized(self, tmp_path):
        segs = [dict(s, translated_text="x") for s in HEALTHY]
        work = _job(tmp_path, tts_done=segs)
        r = audit.audit_job(work)
        assert _has(r, "never synthesized")

    def test_missing_wav_on_disk(self, tmp_path):
        work = tmp_path / "job1"
        work.mkdir()
        (work / "tts_segments").mkdir()
        real = work / "tts_segments" / "seg_0000.wav"
        real.write_bytes(b"\x00" * 4096)
        segs = [dict(HEALTHY[0], translated_text="x", audio_path=str(real)),
                dict(HEALTHY[1], translated_text="x",
                     audio_path=str(work / "tts_segments" / "seg_0001.wav"))]
        (work / "checkpoint_tts_done.json").write_text(
            json.dumps({"segments": segs}), encoding="utf-8")
        r = audit.audit_job(work)
        assert _has(r, "no longer exists")

    def test_stale_wavs_are_info_not_loss(self, tmp_path):
        """188 wavs for 150 segments is confusing but harmless — the kind of
        thing that sends you looking in the wrong place."""
        work = tmp_path / "job1"
        work.mkdir()
        seg_dir = work / "tts_segments"
        seg_dir.mkdir()
        used = seg_dir / "seg_0000.wav"
        used.write_bytes(b"\x00" * 4096)
        (seg_dir / "seg_0099.wav").write_bytes(b"\x00" * 4096)  # leftover
        segs = [dict(HEALTHY[0], translated_text="x", audio_path=str(used))]
        (work / "checkpoint_tts_done.json").write_text(
            json.dumps({"segments": segs}), encoding="utf-8")
        r = audit.audit_job(work)
        assert r["ok"]
        assert any("not referenced" in f["summary"] for f in r["findings"])


class TestAssemblyChecks:
    def test_synthesized_but_unplaced_segment(self, tmp_path):
        work = tmp_path / "job1"
        work.mkdir()
        segs = [dict(s, translated_text="x", audio_path="") for s in HEALTHY]
        (work / "checkpoint_tts_done.json").write_text(
            json.dumps({"segments": segs}), encoding="utf-8")
        (work / "tts_placements.json").write_text(
            json.dumps([{"idx": 0}, {"idx": 1}]), encoding="utf-8")
        r = audit.audit_job(work)
        assert _has(r, "not placed in the dubbed track")

    def test_all_placed_is_clean(self, tmp_path):
        work = tmp_path / "job1"
        work.mkdir()
        segs = [dict(s, translated_text="x", audio_path="") for s in HEALTHY]
        (work / "checkpoint_tts_done.json").write_text(
            json.dumps({"segments": segs}), encoding="utf-8")
        (work / "tts_placements.json").write_text(
            json.dumps([{"idx": i} for i in range(3)]), encoding="utf-8")
        r = audit.audit_job(work)
        assert not any("not placed" in f["summary"] for f in r["findings"])


class TestReportShape:
    def test_report_is_json_serializable(self, tmp_path):
        work = _job(tmp_path, transcribe_done=HEALTHY, transcription_done=HEALTHY)
        json.dumps(audit.audit_job(work))   # must not raise

    def test_stage_rows_carry_counts(self, tmp_path):
        work = _job(tmp_path, transcribe_done=HEALTHY)
        row = audit.audit_job(work)["stages"][0]
        assert row["stage"] == "transcribe"
        assert row["segments"] == 3
        assert row["unique_idx"] == 3
        assert row["words"] == 13

    def test_ok_flag_tracks_loss_findings_only(self, tmp_path):
        work = _job(tmp_path, transcribe_done=HEALTHY, transcription_done=HEALTHY)
        r = audit.audit_job(work)
        assert r["ok"] is (r["counts"]["loss"] == 0)
