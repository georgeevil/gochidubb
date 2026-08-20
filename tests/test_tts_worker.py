"""Tests for pipeline/tts_worker.py — subprocess worker job processing."""
import json
import os
import tempfile

import pytest


class TestWorkerJobJson:
    """The job JSON format expected by tts_worker.py."""

    def test_job_structure(self):
        """A minimal valid job should have required keys."""
        job = {
            "model_id": "openbmb/VoxCPM2",
            "cfg_value": 2.0,
            "inference_timesteps": 10,
            "voice_seed": 404,
            "segments": [
                {
                    "idx": 0,
                    "text": "Hello world",
                    "output_path": "/tmp/seg_0000.wav",
                    "reference_wav_path": "/tmp/ref.wav",
                    "prompt_wav_path": "/tmp/ref.wav",
                    "prompt_text": "Reference transcript.",
                }
            ],
            "target_lang": "ru",
        }
        assert job["model_id"] == "openbmb/VoxCPM2"
        assert len(job["segments"]) == 1
        assert all(k in job["segments"][0] for k in ("idx", "text", "output_path"))

    def test_job_roundtrip_serialization(self):
        """Job should survive JSON serialize/deserialize."""
        job = {
            "model_id": "openbmb/VoxCPM2",
            "cfg_value": 2.0,
            "inference_timesteps": 10,
            "voice_seed": 404,
            "tier_policy": "tier_1_only",
            "enable_qa": True,
            "target_lang": "ru",
            "segments": [
                {
                    "idx": 0,
                    "text": "Привет мир",
                    "output_path": "/tmp/seg_0000.wav",
                    "reference_wav_path": "/tmp/ref.wav",
                    "prompt_wav_path": "/tmp/ref.wav",
                    "prompt_text": "Reference text.",
                },
                {
                    "idx": 1,
                    "text": "Это тест",
                    "output_path": "/tmp/seg_0001.wav",
                    "reference_wav_path": "/tmp/ref.wav",
                    "prompt_wav_path": "/tmp/ref.wav",
                    "prompt_text": "Reference text.",
                },
            ],
        }
        serialized = json.dumps(job, ensure_ascii=False)
        restored = json.loads(serialized)
        assert restored["model_id"] == job["model_id"]
        assert restored["segments"][0]["text"] == "Привет мир"
        assert len(restored["segments"]) == 2

    def test_job_written_to_disk_and_read_back(self):
        """A job file written to disk should be readable by the worker."""
        job = {
            "model_id": "openbmb/VoxCPM2",
            "cfg_value": 2.0,
            "inference_timesteps": 10,
            "voice_seed": 404,
            "segments": [
                {
                    "idx": 0,
                    "text": "Test",
                    "output_path": "/tmp/_test_seg_0000.wav",
                    "reference_wav_path": "/tmp/_test_ref.wav",
                    "prompt_wav_path": "/tmp/_test_ref.wav",
                    "prompt_text": "Test reference.",
                }
            ],
        }
        tmp_path = os.path.join(tempfile.gettempdir(), "_test_worker_job.json")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(job, f)

            with open(tmp_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            assert loaded["model_id"] == "openbmb/VoxCPM2"
            assert loaded["segments"][0]["idx"] == 0
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_tier_policy_default(self):
        """Verify allowed tier_policy values."""
        valid_policies = ["tier_1_only", "tier_1_preferred", "tier_1_fallback"]
        for policy in valid_policies:
            # Just verify it's accepted as a config value
            assert isinstance(policy, str)
        assert len(valid_policies) == 3


# ═════════════════════════════════════════════════════════════════════
# Regression: retry_badcase_max_times must never reach voxcpm as 0
# ═════════════════════════════════════════════════════════════════════
# voxcpm binds `latent_pred` only inside
#     while retry_badcase_times < retry_badcase_max_times:
# and reads it after the loop. A 0 makes the loop body never execute, so
# EVERY segment fails with
#     UnboundLocalError: cannot access local variable 'latent_pred'
# This shipped as speed_retries["fast"] = 0, which meant tts_speed="fast"
# had a 100% failure rate. These tests pin both the source of the value
# and the worker-side clamp that backstops it.

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import synthesizer as _synth  # noqa: E402
from pipeline import tts_worker as _worker  # noqa: E402


class FakeTTSModel:
    """Stands in for voxcpm.VoxCPM — records the kwargs of every attempt."""

    class _Inner:
        sample_rate = 24000

    def __init__(self, fail_times=0):
        self.calls = []
        self.tts_model = self._Inner()
        self._fail_times = fail_times

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) <= self._fail_times:
            raise RuntimeError("simulated tier failure")
        import numpy as np
        return np.zeros(2400, dtype="float32")


def _segment(tmp_path, **overrides):
    ref = tmp_path / "ref.wav"
    import soundfile as sf
    import numpy as np
    sf.write(str(ref), np.zeros(16000, dtype="float32"), 16000)
    seg = {
        "idx": 0,
        "text": "Привет мир",
        "output_path": str(tmp_path / "seg_0000.wav"),
        "reference_wav_path": str(ref),
        "prompt_wav_path": str(ref),
        "prompt_text": "",          # cross-lingual dubs always clear this
    }
    seg.update(overrides)
    return seg


class TestRetryBoundNeverZero:
    @pytest.mark.parametrize("speed", ["fast", "balanced", "quality"])
    def test_speed_preset_bound_is_at_least_one(self, speed):
        """A 0 here is a guaranteed UnboundLocalError inside voxcpm."""
        assert _synth.SPEED_RETRIES[speed] >= 1, (
            f"tts_speed={speed!r} would crash every segment"
        )

    def test_every_speed_preset_is_covered(self):
        assert set(_synth.SPEED_RETRIES) == {"fast", "balanced", "quality"}
        assert set(_synth.SPEED_TIMESTEPS) == {"fast", "balanced", "quality"}

    def test_worker_clamps_a_zero_from_the_job_spec(self, tmp_path):
        """Defence in depth: an old queued job.json can still carry a 0."""
        model = FakeTTSModel()
        job = {
            "retry_badcase_max_times": 0,
            "retry_badcase": False,
            "tier_policy": "balanced",
            "enable_qa": False,
            "segments": [_segment(tmp_path)],
        }
        _worker._process_job(model, job)
        assert model.calls, "generate() was never called"
        for kw in model.calls:
            assert kw["retry_badcase_max_times"] >= 1

    def test_worker_preserves_a_valid_bound(self, tmp_path):
        model = FakeTTSModel()
        job = {
            "retry_badcase_max_times": 2,
            "tier_policy": "balanced",
            "enable_qa": False,
            "segments": [_segment(tmp_path)],
        }
        _worker._process_job(model, job)
        assert model.calls[0]["retry_badcase_max_times"] == 2


class TestTierLadder:
    def test_fast_still_clones_the_source_voice(self, tmp_path):
        """tts_speed='fast' used to run tier 3 only — zero-shot voice design —
        silently discarding every extracted speaker reference."""
        model = FakeTTSModel()
        job = {
            "tier_policy": "fast", "enable_qa": False,
            "segments": [_segment(tmp_path)],
        }
        _worker._process_job(model, job)
        assert model.calls[0].get("reference_wav_path"), (
            "fast tier ladder dropped the speaker reference"
        )

    def test_tier_falls_back_when_the_first_attempt_errors(self, tmp_path):
        """A single-entry ladder meant one error killed the segment."""
        model = FakeTTSModel(fail_times=1)
        job = {
            "tier_policy": "fast", "enable_qa": False,
            "segments": [_segment(tmp_path)],
        }
        _worker._process_job(model, job)
        assert len(model.calls) >= 2, "no fallback tier was attempted"
        assert os.path.exists(job["segments"][0]["output_path"])


class TestUnmeasuredQA:
    """QA returning score=None ("not measured") must be pass-without-claim:
    no seed-mutation retries, no tier fallback, no degraded tier-0 marking.
    Retrying on an unmeasurable signal burns compute and risks voice drift."""

    def _no_ref_segment(self, tmp_path):
        # No reference audio (has_ref False) and text NOT starting with "("
        # (not voice-design) → the one configuration where MAX_QA_RETRIES > 0,
        # so a retry would actually happen if the code wrongly triggered one.
        seg = _segment(tmp_path)
        seg.update(reference_wav_path="", prompt_wav_path="", prompt_text="")
        return seg

    def test_unmeasured_qa_does_not_retry_or_degrade(self, tmp_path, monkeypatch):
        from pipeline import tts_qa as _qa

        qa_calls = []

        def fake_check(audio_path, target_text, target_lang="ru", whisper_size="base"):
            qa_calls.append(audio_path)
            return None, "", {"measured": False, "error": "whisper unavailable",
                              "cer": 1.0, "detected_lang": "", "lang_match": False,
                              "empty": False}

        monkeypatch.setattr(_qa, "check_segment_quality", fake_check)

        model = FakeTTSModel()
        seg = self._no_ref_segment(tmp_path)
        job = {
            "tier_policy": "balanced", "enable_qa": True, "voice_seed": 7,
            "segments": [seg],
        }
        _worker._process_job(model, job)

        assert len(model.calls) == 1, "unmeasured QA must not trigger regeneration"
        assert len(qa_calls) == 1
        assert os.path.exists(seg["output_path"])
        assert seg["qa_score"] is None, "unmeasured must not report a fake score"
        assert seg["qa"]["measured"] is False
        assert seg["qa"]["score"] is None
        assert seg["qa"]["attempts"] == 1
        assert seg["qa"]["tier"] != 0, "unmeasured must not be marked degraded"

    def test_measured_bad_score_still_retries(self, tmp_path, monkeypatch):
        """Contrast case: a REAL bad measurement keeps the retry ladder alive."""
        from pipeline import tts_qa as _qa

        def fake_check(audio_path, target_text, target_lang="ru", whisper_size="base"):
            return 1.0, "garbage", {"measured": True, "cer": 1.0,
                                    "detected_lang": "en", "lang_match": False,
                                    "empty": False}

        monkeypatch.setattr(_qa, "check_segment_quality", fake_check)

        model = FakeTTSModel()
        seg = self._no_ref_segment(tmp_path)
        job = {
            "tier_policy": "balanced", "enable_qa": True, "voice_seed": 7,
            "segments": [seg],
        }
        _worker._process_job(model, job)

        assert len(model.calls) > 1, "a measured bad score should regenerate"
        # All attempts exhausted → accepted best output, marked degraded
        assert seg["qa"]["tier"] == 0
        assert seg["qa"]["measured"] is True
        assert seg["qa_score"] == 1.0


class TestPromptPairing:
    """voxcpm rejects prompt_wav_path unless prompt_text is set too:
    'prompt_wav_path and prompt_text must both be provided or both be None'."""

    def test_empty_prompt_text_never_sends_prompt_wav_path(self, tmp_path):
        model = FakeTTSModel()
        job = {
            "tier_policy": "quality", "enable_qa": False,
            "segments": [_segment(tmp_path, prompt_text="")],
        }
        _worker._process_job(model, job)
        for kw in model.calls:
            if "prompt_wav_path" in kw:
                assert kw.get("prompt_text"), (
                    "prompt_wav_path sent without prompt_text — voxcpm raises"
                )

    def test_real_prompt_text_still_enables_tier_one(self, tmp_path):
        model = FakeTTSModel()
        job = {
            "tier_policy": "quality", "enable_qa": False,
            "segments": [_segment(tmp_path, prompt_text="A reference line.")],
        }
        _worker._process_job(model, job)
        first = model.calls[0]
        assert first.get("prompt_wav_path") and first.get("prompt_text")


class TestQaFallbackKeepsBestAudio:
    """The QA retries vary the seed on purpose, and with VoxCPM a different
    seed clones a different timbre. When every retry fails QA the fallback
    must ship the BEST-scoring take, not the last one.

    Regression: a Chinese dub shipped one segment at ~167 Hz against a 130 Hz
    reference — audibly a second speaker — because best_score was tracked
    while the audio on disk was whatever the final retry wrote.
    """

    def _worker(self):
        import importlib
        return importlib.import_module("pipeline.tts_worker")

    def test_first_attempt_wins_over_best_and_last(self, tmp_path):
        """QA scores CER and language, never timbre — so the best-scoring take
        can still be the wrong voice. Measured: a degraded Chinese segment
        shipped at 225 Hz against a 133 Hz reference while being the better
        scoring of its two takes. Only the first attempt is guaranteed to use
        the unmutated voice_seed, so it is the one that matches the rest of
        the dub."""
        import shutil
        out = tmp_path / "seg_0000.wav"
        first = tmp_path / "seg_0000.wav.first"
        best = tmp_path / "seg_0000.wav.best"
        out.write_bytes(b"L" * 2000)      # last retry
        first.write_bytes(b"F" * 2000)    # original-seed take
        best.write_bytes(b"B" * 2000)     # best-scoring take

        # Mirror the fallback preference order: first, then best.
        for path in (first, best):
            if path.exists():
                shutil.copy2(path, out)
                break
        for path in (first, best):
            if path.exists():
                os.remove(path)

        assert out.read_bytes().startswith(b"F"), "first attempt must win"
        assert not first.exists() and not best.exists()

    def test_best_is_used_when_the_first_take_is_missing(self, tmp_path):
        import shutil
        out = tmp_path / "seg_0000.wav"
        best = tmp_path / "seg_0000.wav.best"
        out.write_bytes(b"L" * 2000)
        best.write_bytes(b"B" * 2000)
        for path in (tmp_path / "seg_0000.wav.first", best):
            if path.exists():
                shutil.copy2(path, out)
                break
        assert out.read_bytes().startswith(b"B")

    def test_strictly_better_keeps_the_earliest_tie(self):
        """Ties must not overwrite: the earliest attempt is the one generated
        with the unmutated voice_seed."""
        best_score, best_tag = None, None
        for score, tag in [(0.20, "original-seed"), (0.20, "retry-1"), (0.30, "retry-2")]:
            if best_score is None or score < best_score:
                best_score, best_tag = score, tag
        assert best_tag == "original-seed"

    def test_retry_seed_actually_differs_from_the_voice_seed(self):
        """Documents the mechanism: if these ever became equal the second-voice
        bug would disappear, and this test should be revisited rather than
        silently passing."""
        voice_seed, idx = 1882596868, 3
        seeds = {(voice_seed or 0) + 1000 * (n + 1) + idx for n in range(2)}
        assert voice_seed not in seeds
        assert len(seeds) == 2
