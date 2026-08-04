"""Shared fixtures and helpers for pipeline tests."""
import json
import os
import tempfile
from typing import Dict, List

import pytest


# ── Sample segment data ──────────────────────────────────────────────

@pytest.fixture
def sample_segments() -> List[Dict]:
    """A realistic set of segments from transcription output."""
    return [
        {"start": 0.5, "end": 2.3, "text": "Hello everyone welcome to my channel.",
         "words": [{"start": 0.5, "end": 0.8, "word": "Hello"},
                    {"start": 0.9, "end": 1.2, "word": "everyone"},
                    {"start": 1.3, "end": 1.5, "word": "welcome"},
                    {"start": 1.6, "end": 1.8, "word": "to"},
                    {"start": 1.9, "end": 2.3, "word": "my channel."}]},
        {"start": 2.5, "end": 3.8, "text": "Today we're talking about AI dubbing.",
         "words": [{"start": 2.5, "end": 2.8, "word": "Today"},
                    {"start": 2.9, "end": 3.1, "word": "we're"},
                    {"start": 3.1, "end": 3.3, "word": "talking"},
                    {"start": 3.4, "end": 3.5, "word": "about"},
                    {"start": 3.6, "end": 3.8, "word": "AI dubbing."}]},
        {"start": 4.0, "end": 6.5, "text": "It's really cool technology that lets you change the language of any video.",
         "words": []},
        {"start": 6.8, "end": 7.2, "text": "Amazing right?"},
        {"start": 7.5, "end": 10.0, "text": "Let me show you how it works step by step."},
    ]


@pytest.fixture
def continuation_segments() -> List[Dict]:
    """Segments where sentences are split at unnatural boundaries."""
    return [
        {"start": 0.0, "end": 1.5, "text": "Who's the most spaz?", "speaker": "SPEAKER_00"},
        {"start": 1.6, "end": 2.0, "text": "Nicky", "speaker": "SPEAKER_00"},
        {"start": 2.2, "end": 3.8, "text": "I mean he just goes absolutely crazy", "speaker": "SPEAKER_00"},
        {"start": 4.0, "end": 5.5, "text": "when he gets on the mat.", "speaker": "SPEAKER_00"},
        {"start": 6.0, "end": 7.2, "text": "But his technique is solid.", "speaker": "SPEAKER_01"},
    ]


@pytest.fixture
def micro_segments() -> List[Dict]:
    """Segments with spurious tiny fragments from pyannote."""
    return [
        {"start": 0.0, "end": 3.0, "text": "This is a normal length segment.", "speaker": "SPEAKER_00"},
        {"start": 3.5, "end": 3.8, "text": "Hi", "speaker": "SPEAKER_00"},
        {"start": 4.0, "end": 7.0, "text": "This is another normal segment.", "speaker": "SPEAKER_00"},
        {"start": 7.2, "end": 7.3, "text": "OK", "speaker": "SPEAKER_01"},
        {"start": 7.5, "end": 10.0, "text": "We continue with more content.", "speaker": "SPEAKER_01"},
    ]


@pytest.fixture
def long_segments() -> List[Dict]:
    """Very long segments that need splitting."""
    return [
        {"start": 0.0, "end": 18.0,
         "text": "Hello and welcome to today's video. In this tutorial we will cover everything you need to know. "
                 "First we start with the basics and then move to advanced topics. Finally we wrap up with a summary.",
         "words": []},
        {"start": 19.0, "end": 22.0, "text": "Short normal segment here.", "words": []},
    ]


@pytest.fixture
def temp_audio_file() -> str:
    """Create a tiny valid WAV file for tests that need audio I/O."""
    import struct
    import wave

    path = os.path.join(tempfile.gettempdir(), "_gochidubb_test_ref.wav")
    if os.path.exists(path):
        return path

    sample_rate = 16000
    duration = 0.5  # seconds
    num_samples = int(sample_rate * duration)

    # Generate a simple sine wave
    import math
    samples = []
    for i in range(num_samples):
        sample = int(32767 * 0.3 * math.sin(2 * math.pi * 440 * i / sample_rate))
        samples.append(sample)

    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for s in samples:
            wf.writeframes(struct.pack('<h', s))

    return path
