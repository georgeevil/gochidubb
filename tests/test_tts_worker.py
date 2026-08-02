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

import sys
from pathlib import Path

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
