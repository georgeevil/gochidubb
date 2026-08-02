"""Tests for pipeline/diagnostics.py — passive checks and the on-demand deep probe."""
import asyncio
from unittest.mock import patch

import pytest

from pipeline import diagnostics as diag
from pipeline.notices import notice


@pytest.fixture(autouse=True)
def _clean_state():
    """Diagnostics keeps process-level caches; don't leak them between tests."""
    diag.clear_runtime()
    diag._last_deep = None
    yield
    diag.clear_runtime()
    diag._last_deep = None


HEALTHY = {
    "python": {"ok": True, "version": "3.11.9"},
    "ffmpeg": {"ok": True, "version": "ffmpeg 8.1"},
    "yt_dlp": {"ok": True, "path": "/usr/bin/yt-dlp"},
    "whisper": {"ok": True, "type": "faster-whisper"},
    "voxcpm": {"ok": True},
    "edge_tts": {"ok": True},
    "pyannote": {"ok": True, "has_token": True},
    "gpu": {"ok": False, "name": "", "vram_gb": 0},
}


def _status(**overrides):
    s = {k: dict(v) for k, v in HEALTHY.items()}
    s.update(overrides)
    return s


def _codes(notices):
    return {n["code"] for n in notices}


class TestPassiveChecks:
    def test_makes_no_network_calls(self, monkeypatch):
        """This runs on the 5s /api/system poll — a network call here would
        hammer huggingface.co and break the app offline."""
        def explode(*a, **kw):
            raise AssertionError("passive_checks must not touch the network")

        monkeypatch.setattr("socket.socket.connect", explode, raising=False)
        monkeypatch.setenv("HF_TOKEN", "hf_abcdefghij0123456789")
        diag.passive_checks(_status())

    def test_healthy_setup_yields_no_problems(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_abcdefghij0123456789")
        with patch.object(diag, "accelerator",
                          return_value={"ok": True, "backend": "torch.cuda", "cuda": True}):
            out = diag.passive_checks(_status())
        assert _codes(out["notices"]) == set()

    def test_missing_ffmpeg_is_an_error_with_platform_steps(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_abcdefghij0123456789")
        out = diag.passive_checks(_status(ffmpeg={"ok": False}))
        n = next(n for n in out["notices"] if n["code"] == "ffmpeg.missing")
        assert n["severity"] == "error"
        assert any("brew install ffmpeg" in s for s in n["remediation"])

    def test_missing_token_disables_diarization(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "")
        monkeypatch.setattr("pipeline.diarizer.HF_TOKEN", "", raising=False)
        from app.config import cfg
        monkeypatch.setattr(cfg, "hf_token", "", raising=False)
        out = diag.passive_checks(_status())
        assert "pyannote.no_token" in _codes(out["notices"])

    def test_non_hf_token_is_flagged(self, monkeypatch):
        """pyannote silently drops tokens that don't start with hf_, so the
        download then fails as if unauthenticated."""
        monkeypatch.setenv("HF_TOKEN", "pyannoteai_key_1234")
        out = diag.passive_checks(_status())
        assert "pyannote.bad_token_format" in _codes(out["notices"])

    def test_edge_tts_only_warns_about_lost_cloning(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_abcdefghij0123456789")
        out = diag.passive_checks(_status(voxcpm={"ok": False, "error": "no module"}))
        n = next(n for n in out["notices"] if n["code"] == "voxcpm.missing")
        assert n["severity"] == "warn", "edge-tts still produces a dub"

    def test_no_tts_engine_at_all_is_an_error(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_abcdefghij0123456789")
        out = diag.passive_checks(_status(voxcpm={"ok": False}, edge_tts={"ok": False}))
        assert "tts.missing" in _codes(out["notices"])

    def test_checklist_covers_every_subsystem_the_ui_shows(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_abcdefghij0123456789")
        ids = {c["id"] for c in diag.passive_checks(_status())["checks"]}
        assert {"python", "ffmpeg", "yt_dlp", "whisper", "tts",
                "pyannote", "hf_token", "gpu"} <= ids


class TestAccelerator:
    def test_reports_mps_rather_than_no_gpu(self):
        """check_gpu() is CUDA-only, which made Apple Silicon look broken."""
        acc = diag.accelerator()
        assert "backend" in acc and "ok" in acc
        if acc.get("mps"):
            assert acc["ok"] is True
            assert "MPS" in acc.get("name", "")


class TestTtsQaNotice:
    def test_warns_when_qa_cannot_load_off_cuda(self):
        ns = diag._tts_qa_notices({"cuda": False})
        assert [n["code"] for n in ns] == ["tts_qa.device_unavailable"]

    def test_silent_on_cuda(self):
        assert diag._tts_qa_notices({"cuda": True}) == []

    def test_says_not_measured_rather_than_good(self):
        """qa=0.00 looks like evidence of quality; it is the absence of it."""
        n = diag._tts_qa_notices({"cuda": False})[0]
        assert "not measured" in n["detail"]
        assert n["severity"] == "warn"


class TestHfProbe:
    def test_no_token_probes_nothing(self):
        assert diag._probe_hf("")["token_valid"] is None

    def test_gated_repo_is_reported_as_gated(self):
        from huggingface_hub.utils import GatedRepoError

        class FakeApi:
            def whoami(self, token=None):
                return {"name": "someone"}

            def model_info(self, repo, token=None):
                if repo == "pyannote/segmentation-3.0":
                    # GatedRepoError requires a real response object; building
                    # it the way the hub does keeps this honest. httpx, not
                    # requests — that's what huggingface_hub uses, and it's
                    # already a declared dependency.
                    import httpx
                    raise GatedRepoError(
                        "Access to model is restricted",
                        response=httpx.Response(
                            403,
                            request=httpx.Request(
                                "GET",
                                "https://huggingface.co/api/models/" + repo),
                        ),
                    )
                return object()

        with patch("huggingface_hub.HfApi", return_value=FakeApi()):
            out = diag._probe_hf("hf_abcdefghij0123456789")
        assert out["token_valid"] is True
        assert out["repos"]["pyannote/segmentation-3.0"]["reason"] == "gated"
        assert out["repos"]["pyannote/speaker-diarization-3.1"]["ok"] is True

    def test_bad_token_is_not_confused_with_gating(self):
        class FakeApi:
            def whoami(self, token=None):
                raise RuntimeError("401 Unauthorized")

        with patch("huggingface_hub.HfApi", return_value=FakeApi()):
            out = diag._probe_hf("hf_bad")
        assert out["token_valid"] is False
        assert out["repos"] == {}

    def test_network_failure_is_not_blamed_on_the_token(self):
        """Telling an offline user to regenerate a working credential is worse
        than saying nothing."""
        class FakeApi:
            def whoami(self, token=None):
                raise ConnectionError("Name or service not known")

        with patch("huggingface_hub.HfApi", return_value=FakeApi()):
            out = diag._probe_hf("hf_abcdefghij0123456789")
        assert out["offline"] is True
        assert out["token_valid"] is None


class TestRuntimeNotices:
    def test_records_and_dedupes_by_code(self):
        diag.record_runtime_notices([notice("a.b", "warn", "first")], job_id="j1")
        diag.record_runtime_notices([notice("a.b", "warn", "second")], job_id="j2")
        got = diag.runtime_notices()
        assert len(got) == 1
        assert got[0]["title"] == "second"
        assert got[0]["job_id"] == "j2"

    def test_appear_in_current_notices(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_abcdefghij0123456789")
        diag.record_runtime_notices([notice("pyannote.fallback_model", "warn", "fell back")])
        assert "pyannote.fallback_model" in _codes(diag.current_notices(_status()))

    def test_clear_targets_specific_codes(self):
        diag.record_runtime_notices([notice("a.b"), notice("c.d")])
        diag.clear_runtime(["a.b"])
        assert [n["code"] for n in diag.runtime_notices()] == ["c.d"]

    def test_revalidation_sets_name_the_codes_a_check_can_retire(self):
        assert "pyannote.fallback_model" in diag.REVALIDATES["hf"]
        assert "translation.unreachable" in diag.REVALIDATES["translation"]


class TestDeepChecks:
    def _run(self, hf_result, backend=(True, ["m1"])):
        async def fake_backend():
            return backend

        with patch.object(diag, "_probe_hf", return_value=hf_result), \
             patch("pipeline.translator.check_ollama", fake_backend), \
             patch.object(diag, "passive_checks",
                          return_value={"checks": [], "notices": [],
                                        "accelerator": {"ok": True}}), \
             patch("pipeline.models.get_system_status", return_value=_status()):
            return asyncio.run(diag.deep_checks("hf_abcdefghij0123456789"))

    def test_gated_repo_becomes_an_actionable_notice(self):
        r = self._run({"token_valid": True, "user": "someone", "offline": False,
                       "repos": {"pyannote/segmentation-3.0": {"ok": False, "reason": "gated"},
                                 "pyannote/speaker-diarization-3.1": {"ok": True}}})
        n = next(n for n in r["notices"] if n["code"] == "pyannote.gated_model")
        assert n["url"] == "https://hf.co/pyannote/segmentation-3.0"
        assert any("accept the user conditions" in s for s in n["remediation"])

    def test_bad_token_maps_to_its_own_code(self):
        r = self._run({"token_valid": False, "offline": False, "repos": {},
                       "error": "401 Unauthorized"})
        assert "pyannote.bad_token" in _codes(r["notices"])
        assert "pyannote.gated_model" not in _codes(r["notices"])

    def test_offline_says_offline(self):
        r = self._run({"token_valid": None, "offline": True, "repos": {},
                       "error": "connection refused"})
        assert "hf.offline" in _codes(r["notices"])

    def test_confirmed_access_retires_a_past_download_failure(self):
        """The whole point of the button: pyannote prints 'private or gated'
        for any HTTP error, so a transient 503 leaves a warning the user
        cannot act on. Asking the Hub directly is what clears it."""
        diag.record_runtime_notices([notice("pyannote.fallback_model", "warn", "fell back")])
        self._run({"token_valid": True, "user": "someone", "offline": False,
                   "repos": {"pyannote/segmentation-3.0": {"ok": True},
                             "pyannote/speaker-diarization-3.1": {"ok": True}}})
        assert diag.runtime_notices() == []

    def test_a_still_gated_repo_does_not_retire_the_warning(self):
        diag.record_runtime_notices([notice("pyannote.fallback_model", "warn", "fell back")])
        self._run({"token_valid": True, "user": "someone", "offline": False,
                   "repos": {"pyannote/segmentation-3.0": {"ok": False, "reason": "gated"}}})
        assert [n["code"] for n in diag.runtime_notices()] == ["pyannote.fallback_model"]

    def test_unreachable_backend_is_an_error(self):
        r = self._run({"token_valid": True, "user": "u", "offline": False, "repos": {}},
                      backend=(False, []))
        assert "translation.unreachable" in _codes(r["notices"])

    def test_backend_with_no_models_is_its_own_problem(self):
        r = self._run({"token_valid": True, "user": "u", "offline": False, "repos": {}},
                      backend=(True, []))
        assert "translation.no_models" in _codes(r["notices"])

    def test_result_is_cached_for_the_next_poll(self):
        self._run({"token_valid": True, "user": "u", "offline": False, "repos": {}})
        assert diag.last_deep() is not None
        assert diag.last_deep()["ran_at"] > 0

    def test_cached_layer_excludes_passive_notices(self, monkeypatch):
        """Caching a passive copy would resurrect problems already fixed —
        the banner would keep reporting an ffmpeg the user just installed."""
        monkeypatch.setenv("HF_TOKEN", "hf_abcdefghij0123456789")

        async def fake_backend():
            return True, ["m1"]

        with patch.object(diag, "_probe_hf",
                          return_value={"token_valid": True, "user": "u",
                                        "offline": False, "repos": {}}), \
             patch("pipeline.translator.check_ollama", fake_backend), \
             patch("pipeline.models.get_system_status",
                   return_value=_status(ffmpeg={"ok": False})):
            r = asyncio.run(diag.deep_checks("hf_abcdefghij0123456789"))

        assert "ffmpeg.missing" in _codes(r["notices"]), "the report shows everything"
        assert "ffmpeg.missing" not in _codes(r["deep_notices"]), "but only network findings persist"

    def test_report_includes_still_valid_runtime_findings(self):
        """The Setup tab renders the deep report, so a runtime problem the
        checks did not disprove must survive the button press — otherwise
        pressing "Run checks" looks like it fixed something."""
        diag.record_runtime_notices([notice("pyannote.inference_failed", "error", "crashed")])
        r = self._run({"token_valid": False, "offline": False, "repos": {},
                       "error": "401"})
        assert "pyannote.inference_failed" in _codes(r["notices"])
