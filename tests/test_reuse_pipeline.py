"""Stage reuse through the real pipeline runner.

app/reuse.py and app/artifact_store.py are tested in isolation elsewhere. What
matters here is the wiring in server.run_pipeline_stages: that reuse is off
unless asked for, that a hit actually skips the handler, and that a second job
in a different language reuses the first job's source-only work — which is the
entire point of the feature.

The stage handlers are replaced with fakes so nothing downloads, transcribes or
synthesizes; the runner, the fingerprints and the store are real.
"""
import asyncio
import pathlib

import pytest

import server
from app import artifact_store, reuse_runtime


class Cfg:
    """Stand-in for app.config.cfg."""
    reuse_enabled = True
    reuse_stages = "download,extract,transcribe,diarize"
    whisper_model = "large-v3"
    voxcpm_cfg = 2.0
    tts_max_stretch = 1.4

    def __init__(self, **over):
        for k, v in over.items():
            setattr(self, k, v)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """A runner wired to temp dirs, fake stages, and a fresh artifact store."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    db = tmp_path / "t.db"
    artifact_store.init_store(db)
    monkeypatch.setattr(artifact_store, "_DB_PATH", db)
    monkeypatch.setattr(server, "OUTPUT_DIR", outputs)
    monkeypatch.setattr(server, "cfg", Cfg())
    monkeypatch.setattr(server, "save_job", lambda job: None)
    monkeypatch.setattr(server, "_save_checkpoint",
                        lambda *a, **k: None)

    ran: list = []

    def fake(stage):
        async def handler(job, work, ctx, update, perf):
            ran.append(stage)
            work.mkdir(parents=True, exist_ok=True)
            if stage == "download":
                # Content follows the source, so two different URLs really do
                # produce different media. With identical bytes, extract would
                # be reused across them — correctly, but confusingly.
                (work / "source_video.mp4").write_bytes(
                    f"video:{ctx.get('source')}".encode())
                ctx["video_path"] = str(work / "source_video.mp4")
                ctx["duration"] = 51.4
            elif stage == "extract":
                # Mirrors the real stage: it writes audio_16k.wav, then
                # denoise and VAD each write ANOTHER file and repoint
                # ctx["audio_16k"] at it. An earlier version of this fake
                # always pointed at audio_16k.wav, which hid a bug where the
                # file a reused job inherited was never copied.
                src = pathlib.Path(ctx["video_path"]).read_bytes()
                (work / "audio_16k.wav").write_bytes(b"audio:" + src)
                current = work / "audio_16k.wav"
                if ctx.get("auto_denoise", True):
                    (work / "audio_16k_clean.wav").write_bytes(b"clean:" + src)
                    current = work / "audio_16k_clean.wav"
                if getattr(server.cfg, "vad_enabled", True):
                    (work / "audio_16k_vad.wav").write_bytes(b"vad:" + src)
                    current = work / "audio_16k_vad.wav"
                ctx["audio_16k"] = str(current)
            elif stage == "transcribe":
                ctx["segments"] = [
                    {"idx": 0, "start": 1.0, "end": 4.0, "text": "Hola mamá",
                     "word_conf_mean": 0.92, "no_speech_prob": 0.02},
                ]
                ctx["source_lang_detected"] = "es"
            elif stage == "diarize":
                refs = work / "speaker_refs"
                refs.mkdir(exist_ok=True)
                (refs / "ref_SPEAKER_00.wav").write_bytes(b"ref")
                ctx["speaker_refs"] = {
                    "SPEAKER_00": str(refs / "ref_SPEAKER_00.wav")}
            elif stage == "translate":
                for s in ctx.get("segments", []):
                    s["translated_text"] = f"[{ctx.get('target_lang')}] {s['text']}"
                (work / "subtitles.srt").write_text("1\n", encoding="utf-8")
                ctx["srt_path"] = str(work / "subtitles.srt")
        return handler

    handlers = {sid: fake(sid) for sid in server.STAGE_ORDER}
    monkeypatch.setattr(server, "STAGE_HANDLERS", handlers)
    return {"outputs": outputs, "ran": ran}


def _run(job_id, ctx, rig, stop_after="diarize"):
    server.jobs[job_id] = {"id": job_id}
    try:
        asyncio.run(server.run_pipeline_stages(
            job_id, ctx, start_stage="download", stop_after=stop_after))
    finally:
        server.jobs.pop(job_id, None)
    return ctx


def _ctx(target_lang="bg", **over):
    base = {"source": "https://example/v", "source_type": "url",
            "target_lang": target_lang, "source_lang": "es",
            "whisper_model": "large-v3", "speaker_mode": "all",
            "auto_denoise": True, "keep_bg": False, "model": "gpt-oss"}
    base.update(over)
    return base


class TestReuseIsOptIn:
    def test_disabled_by_default_in_shipped_config(self):
        """A stale reuse is a silent failure, so this ships off."""
        from app.config import UserConfig
        assert UserConfig().reuse_enabled is False

    def test_nothing_is_reused_when_disabled(self, rig, monkeypatch):
        monkeypatch.setattr(server, "cfg", Cfg(reuse_enabled=False))
        _run("j1", _ctx(), rig)
        rig["ran"].clear()
        _run("j2", _ctx("ru"), rig)
        assert rig["ran"] == ["download", "extract", "transcribe", "diarize"]

    def test_only_listed_stages_are_reused(self, rig, monkeypatch):
        monkeypatch.setattr(server, "cfg", Cfg(reuse_stages="transcribe"))
        _run("j1", _ctx(), rig)
        rig["ran"].clear()
        _run("j2", _ctx("ru"), rig)
        assert "transcribe" not in rig["ran"]
        assert "diarize" in rig["ran"]


class TestCrossLanguageReuse:
    def test_a_redub_into_a_new_language_skips_source_only_stages(self, rig):
        """The feature in one test: ~60% of pipeline time, recovered."""
        _run("j1", _ctx("bg"), rig)
        assert rig["ran"] == ["download", "extract", "transcribe", "diarize"]

        rig["ran"].clear()
        _run("j2", _ctx("ru"), rig)
        assert rig["ran"] == [], "every source-only stage should have been reused"

    def test_reused_context_points_at_the_new_job(self, rig):
        _run("j1", _ctx("bg"), rig)
        ctx2 = _run("j2", _ctx("ru"), rig)
        assert "j2" in ctx2["audio_16k"]
        assert (rig["outputs"] / "j2" / "audio_16k.wav").exists()
        assert "j2" in ctx2["speaker_refs"]["SPEAKER_00"]

    def test_reused_files_are_real_content(self, rig):
        _run("j1", _ctx("bg"), rig)
        _run("j2", _ctx("ru"), rig)
        assert (rig["outputs"] / "j2" / "audio_16k.wav").read_bytes() \
            == (rig["outputs"] / "j1" / "audio_16k.wav").read_bytes()
        assert (rig["outputs"] / "j2" / "source_video.mp4").read_bytes() \
            == (rig["outputs"] / "j1" / "source_video.mp4").read_bytes()
        assert (rig["outputs"] / "j2" / "speaker_refs"
                / "ref_SPEAKER_00.wav").read_bytes() == b"ref"

    def test_transcript_carries_over(self, rig):
        _run("j1", _ctx("bg"), rig)
        ctx2 = _run("j2", _ctx("ru"), rig)
        assert ctx2["segments"][0]["text"] == "Hola mamá"
        assert ctx2["source_lang_detected"] == "es"

    def test_the_job_records_what_it_reused(self, rig):
        """A run that silently skipped work is impossible to debug later."""
        _run("j1", _ctx("bg"), rig)
        server.jobs["j2"] = {"id": "j2"}
        saved = {}
        try:
            asyncio.run(server.run_pipeline_stages(
                "j2", _ctx("ru"), start_stage="download", stop_after="diarize"))
            saved = dict(server.jobs["j2"])
        finally:
            server.jobs.pop("j2", None)
        stages = [r["stage"] for r in saved.get("reused_stages", [])]
        assert stages == ["download", "extract", "transcribe", "diarize"]
        assert all(r["from"] == "j1" for r in saved["reused_stages"])


class TestInputsThatMustInvalidate:
    def test_a_different_whisper_model_re_transcribes(self, rig):
        _run("j1", _ctx("bg"), rig)
        rig["ran"].clear()
        _run("j2", _ctx("bg", whisper_model="medium"), rig)
        assert "transcribe" in rig["ran"]
        # ...but the audio it works from is still reused.
        assert "extract" not in rig["ran"]

    def test_different_source_media_reruns_everything(self, rig):
        _run("j1", _ctx("bg"), rig)
        rig["ran"].clear()
        _run("j2", _ctx("bg", source="https://example/other"), rig)
        assert rig["ran"] == ["download", "extract", "transcribe", "diarize"]

    def test_two_urls_serving_identical_media_share_the_work(self, rig,
                                                             monkeypatch):
        """Content addressing means identity is the bytes, not the URL. Two
        links to the same file legitimately share everything after download."""
        handlers = dict(server.STAGE_HANDLERS)

        async def same_bytes(job, work, ctx, update, perf):
            rig["ran"].append("download")
            work.mkdir(parents=True, exist_ok=True)
            (work / "source_video.mp4").write_bytes(b"identical")
            ctx["video_path"] = str(work / "source_video.mp4")
            ctx["duration"] = 51.4
        handlers["download"] = same_bytes
        monkeypatch.setattr(server, "STAGE_HANDLERS", handlers)

        _run("j1", _ctx("bg"), rig)
        rig["ran"].clear()
        _run("j2", _ctx("bg", source="https://mirror/same"), rig)
        assert rig["ran"] == ["download"], "only the fetch should repeat"

    def test_toggling_denoise_re_extracts(self, rig):
        _run("j1", _ctx("bg"), rig)
        rig["ran"].clear()
        _run("j2", _ctx("bg", auto_denoise=False), rig)
        assert "extract" in rig["ran"]

    def test_a_stage_version_bump_retires_the_cache(self, rig, monkeypatch):
        from app import reuse as app_reuse
        _run("j1", _ctx("bg"), rig)
        monkeypatch.setitem(app_reuse.STAGE_VERSIONS, "transcribe",
                            app_reuse.STAGE_VERSIONS["transcribe"] + 1)
        rig["ran"].clear()
        _run("j2", _ctx("bg"), rig)
        assert "transcribe" in rig["ran"]


class TestQualityGating:
    def test_a_low_confidence_transcript_is_not_reused(self, rig, monkeypatch):
        """Recorded when written, refused when read."""
        handlers = dict(server.STAGE_HANDLERS)
        original = handlers["transcribe"]

        async def poor(job, work, ctx, update, perf):
            await original(job, work, ctx, update, perf)
            ctx["segments"][0]["word_conf_mean"] = 0.05
        handlers["transcribe"] = poor
        monkeypatch.setattr(server, "STAGE_HANDLERS", handlers)

        _run("j1", _ctx("bg"), rig)
        rig["ran"].clear()
        _run("j2", _ctx("ru"), rig)
        assert "transcribe" in rig["ran"], "a bad transcript must not spread"
        assert "extract" not in rig["ran"], "the audio is still fine to reuse"


class TestFailureModes:
    def test_a_deleted_source_job_falls_back_to_running(self, rig):
        import shutil
        _run("j1", _ctx("bg"), rig)
        shutil.rmtree(rig["outputs"] / "j1")
        rig["ran"].clear()
        _run("j2", _ctx("ru"), rig)
        assert rig["ran"] == ["download", "extract", "transcribe", "diarize"]

    def test_a_broken_store_never_fails_a_job(self, rig, monkeypatch):
        """Caching is an optimisation. An optimisation that can fail a job is
        worse than no optimisation, so every store call is guarded."""
        def boom(*a, **k):
            raise RuntimeError("store is on fire")
        monkeypatch.setattr(reuse_runtime.artifact_store, "lookup", boom)
        monkeypatch.setattr(reuse_runtime.artifact_store, "record", boom)
        _run("j1", _ctx("bg"), rig)
        assert rig["ran"] == ["download", "extract", "transcribe", "diarize"]

    def test_a_context_that_cannot_be_fingerprinted_just_runs(self, rig,
                                                              monkeypatch):
        monkeypatch.setattr(reuse_runtime, "build_fingerprint_inputs",
                            lambda *a, **k: (_ for _ in ()).throw(ValueError("x")))
        _run("j1", _ctx("bg"), rig)
        assert rig["ran"] == ["download", "extract", "transcribe", "diarize"]


class TestReviewRegressions:
    """One per defect a review found in the first cut of this feature. Each
    was a silent wrong-output bug, not a crash."""

    def test_the_audio_a_reused_job_inherits_actually_exists(self, rig):
        """extract repoints ctx["audio_16k"] at audio_16k_clean.wav or
        audio_16k_vad.wav. Recording a hard-coded file list meant the reusing
        job inherited a path to a file that was never copied."""
        _run("j1", _ctx("bg"), rig)
        ctx2 = _run("j2", _ctx("ru"), rig)
        assert pathlib.Path(ctx2["audio_16k"]).exists()
        assert "j2" in ctx2["audio_16k"]

    def test_the_same_holds_with_vad_off(self, rig, monkeypatch):
        monkeypatch.setattr(server, "cfg", Cfg(vad_enabled=False))
        _run("j1", _ctx("bg"), rig)
        ctx2 = _run("j2", _ctx("ru"), rig)
        assert ctx2["audio_16k"].endswith("audio_16k_clean.wav")
        assert pathlib.Path(ctx2["audio_16k"]).exists()

    def test_turning_vad_off_re_extracts(self, rig, monkeypatch):
        """VAD changes both the audio and which file ctx names, so reusing
        across the setting would transcribe audio the job asked not to have."""
        _run("j1", _ctx("bg"), rig)
        monkeypatch.setattr(server, "cfg", Cfg(vad_enabled=False))
        rig["ran"].clear()
        _run("j2", _ctx("bg"), rig)
        assert "extract" in rig["ran"]

    def test_an_uploaded_voice_is_not_overwritten_by_cached_refs(self, rig,
                                                                 tmp_path):
        """A job with an uploaded reference must not silently inherit speaker
        references cut from the video."""
        ref = tmp_path / "myvoice.wav"
        ref.write_bytes(b"my voice")
        _run("j1", _ctx("bg"), rig)
        rig["ran"].clear()
        _run("j2", _ctx("bg", reference_audio=str(ref)), rig)
        assert "diarize" in rig["ran"]

    def test_skipping_diarization_is_part_of_the_key(self, rig):
        _run("j1", _ctx("bg"), rig)
        rig["ran"].clear()
        _run("j2", _ctx("bg", skip_diarization=True), rig)
        assert "diarize" in rig["ran"]

    def test_changing_the_configured_translation_model_re_translates(
            self, rig, monkeypatch):
        """_stage_translate falls back to cfg.translation_model, so a job that
        does not override it must still be keyed on what it actually used."""
        cfg_a = Cfg(reuse_stages="translate", translation_model="gpt-oss")
        monkeypatch.setattr(server, "cfg", cfg_a)
        ctx = _ctx("bg")
        ctx.pop("model")
        _run("j1", dict(ctx), rig, stop_after="translate")
        monkeypatch.setattr(server, "cfg",
                            Cfg(reuse_stages="translate", translation_model="qwen"))
        rig["ran"].clear()
        _run("j2", dict(ctx), rig, stop_after="translate")
        assert "translate" in rig["ran"]

    def test_a_reused_translate_still_leaves_an_srt_on_disk(self, rig,
                                                            monkeypatch):
        monkeypatch.setattr(server, "cfg",
                            Cfg(reuse_stages="download,extract,transcribe,"
                                             "diarize,translate"))
        _run("j1", _ctx("bg"), rig, stop_after="translate")
        rig["ran"].clear()
        server.jobs["j2"] = {"id": "j2"}
        try:
            asyncio.run(server.run_pipeline_stages(
                "j2", _ctx("bg"), start_stage="download",
                stop_after="translate"))
            saved = dict(server.jobs["j2"])
        finally:
            server.jobs.pop("j2", None)
        assert "translate" not in rig["ran"], "should have been reused"
        assert (rig["outputs"] / "j2" / "subtitles.srt").exists()
        assert saved.get("srt_url") == "/outputs/j2/subtitles.srt"

    def test_a_reused_download_still_publishes_the_duration(self, rig):
        """job["duration"] feeds the publisher's duplicate check; a skipped
        stage never runs its own update()."""
        _run("j1", _ctx("bg"), rig)
        server.jobs["j2"] = {"id": "j2"}
        try:
            asyncio.run(server.run_pipeline_stages(
                "j2", _ctx("ru"), start_stage="download", stop_after="diarize"))
            saved = dict(server.jobs["j2"])
        finally:
            server.jobs.pop("j2", None)
        assert saved.get("duration") == 51.4
        assert saved.get("segment_count") == 1
        assert saved.get("speaker_count") == 1
