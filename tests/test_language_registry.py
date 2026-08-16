"""Language registry consistency — one list to rule them all.

GET /api/languages is derived from EdgeTTSFallback.VOICE_MAP; the batch
endpoints (Quick Test / Showcase / Redub) validate target codes against
server._QUICK_TEST_KNOWN_LANGS. These two must never drift, or a language
the API advertises gets rejected at submission time — which is exactly
what used to happen to "bg" (Bulgarian).
"""
import server

from pipeline.synthesizer import EdgeTTSFallback


def test_batch_endpoints_accept_every_advertised_language():
    assert set(EdgeTTSFallback.VOICE_MAP) <= server._QUICK_TEST_KNOWN_LANGS


def test_bulgarian_is_a_first_class_target():
    # The regression this file pins: bg was in the voice map (so
    # /api/languages listed it) but missing from the UI dropdown and the
    # batch-endpoint validation set, so Bulgarian dubs couldn't be started.
    assert "bg" in EdgeTTSFallback.VOICE_MAP
    assert "bg" in server._QUICK_TEST_KNOWN_LANGS


def test_edge_only_targets_still_have_voices():
    # Languages routed around the cloning engines (they are not in VoxCPM2's
    # training coverage) MUST have an edge-tts voice, or the dub has no TTS.
    assert server._EDGE_ONLY_TARGET_LANGS <= set(EdgeTTSFallback.VOICE_MAP)