"""Tests for app/reuse.py — fingerprints and quality gates.

The dangerous failure this feature can produce is a *stale reuse*: serving one
job's work for another job that would have computed something different. Most
of these tests are about that — what must change a fingerprint, and what must
not.
"""
import json

import pytest

from app import reuse


class TestHashFile:
    def test_same_content_same_hash(self, tmp_path):
        a, b = tmp_path / "a.bin", tmp_path / "b.bin"
        a.write_bytes(b"hello world"), b.write_bytes(b"hello world")
        assert reuse.hash_file(str(a)) == reuse.hash_file(str(b))

    def test_different_content_different_hash(self, tmp_path):
        a, b = tmp_path / "a.bin", tmp_path / "b.bin"
        a.write_bytes(b"hello world"), b.write_bytes(b"hello worlds")
        assert reuse.hash_file(str(a)) != reuse.hash_file(str(b))

    def test_identity_is_content_not_path(self, tmp_path):
        """uploads/video.mp4 gets reused for different uploads; keying on the
        path would serve one video's transcript for another's audio."""
        p = tmp_path / "video.mp4"
        p.write_bytes(b"first upload")
        first = reuse.hash_file(str(p))
        p.write_bytes(b"second upload")
        assert reuse.hash_file(str(p)) != first

    def test_missing_file_hashes_to_empty(self, tmp_path):
        assert reuse.hash_file(str(tmp_path / "nope.bin")) == ""

    def test_prefix_hashing_still_notices_a_truncation(self, tmp_path):
        """Size is mixed in, so a file cut short past the prefix is caught."""
        a, b = tmp_path / "a.bin", tmp_path / "b.bin"
        a.write_bytes(b"x" * 4096)
        b.write_bytes(b"x" * 2048)
        assert reuse.hash_file(str(a), max_bytes=1024) != \
               reuse.hash_file(str(b), max_bytes=1024)


class TestHashValue:
    def test_key_order_does_not_change_the_hash(self):
        assert reuse.hash_value({"a": 1, "b": 2}) == reuse.hash_value({"b": 2, "a": 1})

    def test_different_values_differ(self):
        assert reuse.hash_value({"a": 1}) != reuse.hash_value({"a": 2})

    def test_survives_unserializable_input(self):
        assert reuse.hash_value(object())


class TestHashSegments:
    def _segs(self, **over):
        base = [{"start": 1.0, "end": 2.0, "text": "Hola mamá",
                 "avg_logprob": -0.2, "speaker": "SPEAKER_00",
                 "word_conf_mean": 0.9}]
        base[0].update(over)
        return base

    def test_confidence_and_speaker_are_ignored(self):
        """Two transcripts with the same words at the same times produce the
        same translation, so the noise must not defeat the cache."""
        a = reuse.hash_segments(self._segs())
        b = reuse.hash_segments(self._segs(avg_logprob=-0.9, speaker="SPEAKER_03",
                                           word_conf_mean=0.4))
        assert a == b

    def test_text_changes_the_hash(self):
        assert reuse.hash_segments(self._segs()) != \
               reuse.hash_segments(self._segs(text="Hola papá"))

    def test_timing_changes_the_hash(self):
        assert reuse.hash_segments(self._segs()) != \
               reuse.hash_segments(self._segs(end=3.0))

    def test_empty_is_stable(self):
        assert reuse.hash_segments([]) == reuse.hash_segments(None)


class TestStageFingerprint:
    def _ctx(self, **over):
        base = {"audio_fingerprint": "aaa", "whisper_model": "large-v3",
                "source_lang": "es"}
        base.update(over)
        return base

    def test_same_inputs_same_fingerprint(self):
        assert reuse.stage_fingerprint("transcribe", self._ctx()) == \
               reuse.stage_fingerprint("transcribe", self._ctx())

    def test_whisper_model_is_part_of_the_key(self):
        assert reuse.stage_fingerprint("transcribe", self._ctx()) != \
               reuse.stage_fingerprint("transcribe", self._ctx(whisper_model="medium"))

    def test_audio_content_is_part_of_the_key(self):
        assert reuse.stage_fingerprint("transcribe", self._ctx()) != \
               reuse.stage_fingerprint("transcribe", self._ctx(audio_fingerprint="bbb"))

    def test_stage_version_retires_cached_work(self, monkeypatch):
        """The silent failure this guards: upgrading faster-whisper without
        bumping the version would serve transcripts from the old one forever."""
        before = reuse.stage_fingerprint("transcribe", self._ctx())
        monkeypatch.setitem(reuse.STAGE_VERSIONS, "transcribe",
                            reuse.STAGE_VERSIONS["transcribe"] + 1)
        assert reuse.stage_fingerprint("transcribe", self._ctx()) != before

    def test_two_stages_never_share_a_fingerprint(self):
        ctx = {"audio_fingerprint": "aaa", "whisper_model": "m",
               "source_lang": "es", "speaker_mode": "all",
               "diarization_model": "d"}
        assert reuse.stage_fingerprint("transcribe", ctx) != \
               reuse.stage_fingerprint("diarize", ctx)

    def test_missing_content_hash_refuses_to_key(self):
        """An unreadable source must not be filed under a fingerprint that
        does not describe it."""
        assert reuse.stage_fingerprint("transcribe", self._ctx(audio_fingerprint="")) is None

    def test_absent_option_differs_from_a_set_one(self):
        """"the caller forgot auto_denoise" must not collide with
        "auto_denoise was off"."""
        a = reuse.stage_fingerprint("extract", {"source_fingerprint": "s"})
        b = reuse.stage_fingerprint("extract", {"source_fingerprint": "s",
                                                "auto_denoise": False})
        assert a != b

    def test_uncacheable_stage_returns_none(self):
        assert reuse.stage_fingerprint("assemble", {"anything": 1}) is None

    def test_translate_key_covers_model_language_and_glossary(self):
        base = {"segments_fingerprint": "s", "target_lang": "bg",
                "model": "m", "context_hint": "", "glossary_fingerprint": "g"}
        fp = reuse.stage_fingerprint("translate", base)
        for change in ({"target_lang": "ru"}, {"model": "other"},
                       {"glossary_fingerprint": "g2"}, {"context_hint": "hint"}):
            assert reuse.stage_fingerprint("translate", {**base, **change}) != fp

    def test_assemble_and_merge_are_not_cacheable(self):
        """They are 3.2% of runtime and the stages you actually re-tune."""
        assert "assemble" not in reuse.CACHEABLE_STAGES
        assert "merge" not in reuse.CACHEABLE_STAGES

    def test_every_cacheable_stage_has_a_version(self):
        for stage in reuse.CACHEABLE_STAGES:
            assert reuse.STAGE_VERSIONS.get(stage), stage


class TestEvaluateQuality:
    def test_transcribe_summarises_confidence(self):
        ctx = {"segments": [
            {"word_conf_mean": 0.9, "no_speech_prob": 0.1},
            {"word_conf_mean": 0.5, "no_speech_prob": 0.9},
        ]}
        q = reuse.evaluate_quality("transcribe", ctx)
        assert q["word_conf_mean"] == pytest.approx(0.7)
        assert q["no_speech_ratio"] == pytest.approx(0.5)

    def test_translate_counts_untranslated_segments(self):
        ctx = {"segments": [
            {"text": "Hola", "translated_text": "Здравей"},
            {"text": "Gracias", "translated_text": "Gracias"},   # fell back
        ]}
        assert reuse.evaluate_quality("translate", ctx)["fallback_ratio"] == 0.5

    def test_tts_counts_missing_audio_and_failed_qa(self):
        ctx = {"segments": [
            {"audio_path": "a.wav", "qa_score": 0.1},
            {"audio_path": "b.wav", "qa_score": 0.9},
            {"audio_path": None},
        ]}
        q = reuse.evaluate_quality("tts", ctx)
        assert q["missing_audio"] == 1
        assert q["failed_qa_ratio"] == pytest.approx(0.5)

    def test_unmeasurable_stage_is_empty(self):
        assert reuse.evaluate_quality("download", {"segments": []}) == {}

    def test_quality_is_json_serialisable(self):
        """It is stored in SQLite as JSON."""
        ctx = {"segments": [{"word_conf_mean": 0.8, "no_speech_prob": 0.1}]}
        json.dumps(reuse.evaluate_quality("transcribe", ctx))


class TestPassesGate:
    def test_good_transcript_passes(self):
        ok, _ = reuse.passes_gate("transcribe", {"word_conf_mean": 0.9,
                                                 "no_speech_ratio": 0.1})
        assert ok

    def test_low_confidence_transcript_is_refused(self):
        ok, why = reuse.passes_gate("transcribe", {"word_conf_mean": 0.2})
        assert not ok and "confidence" in why

    def test_mostly_non_speech_transcript_is_refused(self):
        ok, why = reuse.passes_gate("transcribe", {"no_speech_ratio": 0.9})
        assert not ok and "non-speech" in why

    def test_short_speaker_reference_is_refused(self):
        """A 0.4s reference clones badly, so reusing it spreads the problem."""
        ok, why = reuse.passes_gate("diarize", {"min_ref_sec": 0.4})
        assert not ok and "reference" in why

    def test_any_untranslated_segment_refuses_by_default(self):
        ok, why = reuse.passes_gate("translate", {"fallback_ratio": 0.1})
        assert not ok and "untranslated" in why

    def test_weak_semantic_score_is_refused(self):
        ok, why = reuse.passes_gate("translate", {"fallback_ratio": 0.0,
                                                  "semantic_score": 0.3})
        assert not ok and "semantic" in why

    def test_tts_with_missing_audio_is_refused(self):
        ok, why = reuse.passes_gate("tts", {"missing_audio": 2})
        assert not ok and "no audio" in why

    def test_unmeasured_quality_passes(self):
        """Refusing everything unmeasured would disable the cache on the first
        stage that cannot score itself."""
        assert reuse.passes_gate("transcribe", {})[0]
        assert reuse.passes_gate("transcribe", None)[0]

    def test_stage_without_a_gate_passes(self):
        assert reuse.passes_gate("download", {"anything": 1})[0]

    def test_thresholds_are_overridable(self):
        quality = {"word_conf_mean": 0.3}
        assert not reuse.passes_gate("transcribe", quality)[0]
        assert reuse.passes_gate("transcribe", quality,
                                 {"transcribe_min_word_conf": 0.1})[0]

    def test_gates_from_config_reads_user_settings(self):
        class Cfg:
            reuse_transcribe_min_word_conf = 0.8
        g = reuse.gates_from_config(Cfg())
        assert g["transcribe_min_word_conf"] == 0.8
        # Unset gates keep their defaults.
        assert g["diarize_min_ref_sec"] == reuse.DEFAULT_GATES["diarize_min_ref_sec"]


class TestEmptyTranslationsAreUntranslated:
    """The gate exists to stop a bad translation spreading. Counting only
    "translation == source" let a run where the model returned blanks record
    a perfect score and sail through it."""

    def test_blank_translation_counts_as_a_fallback(self):
        ctx = {"segments": [
            {"text": "Hola", "translated_text": "Здравей"},
            {"text": "Gracias", "translated_text": ""},
            {"text": "Adiós", "translated_text": "   "},
            {"text": "Mamá"},                       # key absent entirely
        ]}
        q = reuse.evaluate_quality("translate", ctx)
        assert q["fallback_ratio"] == pytest.approx(0.75)
        assert not reuse.passes_gate("translate", q)[0]

    def test_an_empty_source_segment_is_not_a_fallback(self):
        """Silent segments legitimately have no translation."""
        ctx = {"segments": [{"text": "  ", "translated_text": ""},
                            {"text": "Hola", "translated_text": "Здравей"}]}
        assert reuse.evaluate_quality("translate", ctx)["fallback_ratio"] == 0.0
