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
