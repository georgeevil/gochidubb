"""Tests for the download-rescue path (CLD-242): a failed download attaches
a structured hint to the job, and attach_source's helpers let a manually
downloaded file resume the pipeline from 'extract'.
"""
import asyncio
import shutil
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

from pipeline.downloader import DownloadFailed  # noqa: E402


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


# ── download_hint capture in the pipeline runner ─────────────────────

class TestHintCapture:
    def test_download_failed_hint_lands_on_the_job(self, job_env, monkeypatch):
        hint = {"failure_class": "bot-check", "summary": "s", "detail": "d",
                "commands": [{"label": "l", "command": "c"}]}

        async def boom(job, work, ctx, update, perf):
            raise DownloadFailed("YouTube download failed: x", hint=hint)

        monkeypatch.setitem(server.STAGE_HANDLERS, "download", boom)
        asyncio.run(server.run_pipeline_stages(
            job_env.job_id, {}, start_stage="download", stop_after="download"))

        assert job_env.job["download_hint"] == hint
        assert job_env.job["last_error"]["stage"] == "download"
        assert job_env.job["status"] == "error"

    def test_plain_failure_attaches_no_hint(self, job_env, monkeypatch):
        async def boom(job, work, ctx, update, perf):
            raise RuntimeError("something else broke")

        monkeypatch.setitem(server.STAGE_HANDLERS, "download", boom)
        asyncio.run(server.run_pipeline_stages(
            job_env.job_id, {}, start_stage="download", stop_after="download"))

        assert "download_hint" not in job_env.job
        assert job_env.job["status"] == "error"


class TestClearJobError:
    def test_pops_download_hint(self):
        job = {"id": "x", "error": "boom", "last_error": {"stage": "download"},
               "failed_stage": "download",
               "download_hint": {"failure_class": "unknown"}}
        server._clear_job_error(job)
        assert "download_hint" not in job
        assert "error" not in job
        assert "last_error" not in job


# ── _fresh_run_ctx (regression guard for the retry_stage refactor) ───

class TestFreshRunCtx:
    # Exactly the keys the pre-refactor retry_stage index-0 branch built.
    EXPECTED_KEYS = {
        "source", "source_lang", "target_lang", "model", "keep_bg",
        "whisper_model", "reference_audio", "speaker_mode", "context_hint",
        "voice_style", "voice_preset", "tts_speed", "auto_denoise",
    }

    def test_produces_every_pre_refactor_key(self):
        ctx = server._fresh_run_ctx({"id": "j1"})
        assert set(ctx) == self.EXPECTED_KEYS

    def test_defaults_match_the_pre_refactor_branch(self):
        ctx = server._fresh_run_ctx({"id": "j1"})
        assert ctx["source"] == ""
        assert ctx["source_lang"] == "auto"
        assert ctx["target_lang"] == "ru"
        assert ctx["model"] == server.cfg.translation_model
        assert ctx["keep_bg"] is False
        assert ctx["whisper_model"] == server.cfg.whisper_model
        assert ctx["reference_audio"] == ""
        assert ctx["speaker_mode"] == "main"
        assert ctx["voice_preset"] == "auto"
        assert ctx["tts_speed"] == "balanced"
        assert ctx["auto_denoise"] is True

    def test_job_values_carry_through(self):
        job = {"id": "j1", "source": "https://youtu.be/abc",
               "target_lang": "de", "keep_bg": 1, "model": "aya",
               "speaker_mode": "all"}
        ctx = server._fresh_run_ctx(job)
        assert ctx["source"] == "https://youtu.be/abc"
        assert ctx["target_lang"] == "de"
        assert ctx["keep_bg"] is True
        assert ctx["model"] == "aya"
        assert ctx["speaker_mode"] == "all"


# ── _validate_attached_video ─────────────────────────────────────────

@pytest.mark.skipif(shutil.which("ffprobe") is None,
                    reason="ffprobe not installed")
class TestValidateAttachedVideo:
    def test_rejects_audio_only_file(self, temp_audio_file):
        # A valid WAV has a duration but no video stream.
        why = server._validate_attached_video(Path(temp_audio_file))
        assert why is not None
        assert "video stream" in why

    def test_rejects_a_text_file(self, tmp_path):
        p = tmp_path / "not_a_video.mp4"
        p.write_text("this is not a video")
        assert server._validate_attached_video(p) is not None


# ── attach_source route (direct await, no HTTP) ──────────────────────

def _make_tiny_mp4(path: Path) -> bool:
    """Render a 0.5s test-pattern mp4. False when ffmpeg can't."""
    import subprocess
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i",
             "testsrc=duration=0.5:size=64x64:rate=10",
             "-pix_fmt", "yuv420p", str(path)],
            check=True, capture_output=True, timeout=30)
        return True
    except Exception:
        return False


@pytest.mark.skipif(shutil.which("ffmpeg") is None
                    or shutil.which("ffprobe") is None,
                    reason="ffmpeg/ffprobe not installed")
class TestAttachSourceRoute:
    @pytest.fixture
    def rescue_env(self, job_env, tmp_path, monkeypatch):
        """job_env plus an isolated UPLOAD_DIR and a captured enqueue."""
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()
        monkeypatch.setattr(server, "UPLOAD_DIR", upload_dir)
        enqueued = []

        async def fake_enqueue(job_id, pipeline_args):
            enqueued.append((job_id, pipeline_args))

        monkeypatch.setattr(server, "enqueue_job", fake_enqueue)
        job_env.upload_dir = upload_dir
        job_env.enqueued = enqueued
        return job_env

    def _upload(self, path: Path):
        from fastapi import UploadFile
        return UploadFile(open(path, "rb"), filename=path.name)

    def test_busy_job_is_refused_with_409(self, rescue_env, tmp_path):
        rescue_env.job["status"] = "transcribing"
        clip = tmp_path / "clip.mp4"
        assert _make_tiny_mp4(clip)
        r = asyncio.run(server.attach_source(
            rescue_env.job_id, self._upload(clip)))
        assert r.status_code == 409
        assert not rescue_env.enqueued
        assert not (rescue_env.work / "source_video.mp4").exists()

    def test_unknown_job_is_404(self, rescue_env, tmp_path):
        clip = tmp_path / "clip.mp4"
        assert _make_tiny_mp4(clip)
        r = asyncio.run(server.attach_source("nope", self._upload(clip)))
        assert r.status_code == 404

    def test_invalid_file_is_rejected_and_temp_cleaned(self, rescue_env,
                                                       tmp_path):
        rescue_env.job["status"] = "error"
        bogus = tmp_path / "bogus.mp4"
        bogus.write_text("this is not a video")
        r = asyncio.run(server.attach_source(
            rescue_env.job_id, self._upload(bogus)))
        assert r.status_code == 400
        # The spooled temp copy must not linger in UPLOAD_DIR.
        assert list(rescue_env.upload_dir.iterdir()) == []
        assert not rescue_env.enqueued
        assert rescue_env.job["status"] == "error"  # untouched

    def test_valid_video_resumes_from_extract(self, rescue_env, tmp_path):
        job = rescue_env.job
        job.update(status="error", error="boom",
                   failed_stage="download",
                   last_error={"stage": "download"},
                   download_hint={"failure_class": "bot-check"},
                   source="https://youtu.be/abc")
        clip = tmp_path / "clip.mp4"
        assert _make_tiny_mp4(clip)

        r = asyncio.run(server.attach_source(
            rescue_env.job_id, self._upload(clip)))

        assert r["ok"] is True
        assert r["resumed_from"] == "extract"
        assert r["duration"] == pytest.approx(0.5, abs=0.2)
        # File installed where the download stage would have put it.
        dest = rescue_env.work / "source_video.mp4"
        assert dest.exists() and dest.stat().st_size > 0
        assert list(rescue_env.upload_dir.iterdir()) == []
        # Error state cleared, provenance kept, status queued.
        assert "error" not in job and "download_hint" not in job
        assert job["source"] == "https://youtu.be/abc"
        assert job["rescued_with_upload"] == "clip.mp4"
        assert job["status"] == "queued"
        # The download checkpoint exists so retry_stage("extract") works.
        cp = server._load_checkpoint(rescue_env.job_id, "download_done")
        assert cp and cp["video_path"] == str(dest)
        # Resumed through the same __stage_retry__ path retry_stage uses.
        (jid, args), = rescue_env.enqueued
        assert jid == rescue_env.job_id
        retry = args["__stage_retry__"]
        assert retry["start_stage"] == "extract"
        assert retry["stop_after"] == ""
        assert retry["ctx"]["video_path"] == str(dest)
        assert retry["ctx"]["wizard_mode"] == "auto"

    def test_second_call_while_queued_is_409_no_double_enqueue(
            self, rescue_env, tmp_path):
        rescue_env.job["status"] = "error"
        clip = tmp_path / "clip.mp4"
        assert _make_tiny_mp4(clip)
        r1 = asyncio.run(server.attach_source(
            rescue_env.job_id, self._upload(clip)))
        assert r1["ok"] is True and rescue_env.job["status"] == "queued"
        # "queued" is in _BUSY_STATUSES, so a rapid second call is refused
        # instead of enqueueing the same job twice.
        r2 = asyncio.run(server.attach_source(
            rescue_env.job_id, self._upload(clip)))
        assert r2.status_code == 409
        assert len(rescue_env.enqueued) == 1
