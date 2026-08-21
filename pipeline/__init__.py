"""GoChiDUBB Studio Pipeline - modular dubbing components.

Imports are **lazy** (PEP 562). `from pipeline import transcribe` and
`pipeline.transcribe` both work exactly as before, but the name is resolved
on first access rather than at package import.

This is not a style preference. The package used to import all nine
submodules eagerly, so importing *any* of them pulled in the heaviest:

    tests/test_segment_post.py -> pipeline/__init__.py
                               -> pipeline/transcriber.py -> import torch

`pipeline/segment_post.py` is pure string and timing arithmetic with no
dependencies of its own, but reaching it required torch to be installed. In a
clean environment that meant 22 of 36 test files failed to *collect* and zero
tests ran — which is why the test suite could not be gated on in CI. See
CLD-225.

The eager list also made the failure invisible in the other direction: a
missing optional dependency in a submodule nobody was using still broke
`import pipeline`.
"""
from typing import TYPE_CHECKING

# name -> submodule it lives in. This is the whole public surface; keep it in
# sync with the re-exports below and the __all__ at the bottom.
_LAZY = {
    "download_video": "downloader",
    "extract_audio": "audio",
    "extract_audio_hq": "audio",
    "separate_background": "audio",
    "get_duration": "audio",
    "transcribe": "transcriber",
    "diarize_speakers": "diarizer",
    "assign_speakers_to_segments": "diarizer",
    "extract_speaker_audio": "diarizer",
    "translate_segments": "translator",
    "check_ollama": "translator",
    "ollama_pull_stream": "translator",
    "BaseTTSEngine": "synthesizer",
    "VoxCPMSynthesizer": "synthesizer",
    "F5TTSEngine": "synthesizer",
    "CosyVoiceEngine": "synthesizer",
    "EdgeTTSFallback": "synthesizer",
    "assemble_dubbed_audio": "assembler",
    "merge_audio_video": "assembler",
    "write_srt": "assembler",
    "format_srt_time": "assembler",
    "get_system_status": "models",
    "MODEL_CATALOG": "models",
    "apply_vad_filter": "vad",
    "get_speech_timestamps": "vad",
}


def __getattr__(name: str):
    """Resolve a re-exported name on first access (PEP 562)."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    value = getattr(importlib.import_module(f".{module}", __name__), name)
    globals()[name] = value      # cache, so this runs once per name
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


# Type checkers and IDEs don't execute __getattr__, so give them the real
# imports. TYPE_CHECKING is False at runtime, so this costs nothing.
#
# F401 is ignored for this file in ruff.toml: these are re-exports resolved
# lazily at runtime via __getattr__, and __all__ is computed from _LAZY, so a
# linter cannot connect these names to the public surface they describe.
if TYPE_CHECKING:  # pragma: no cover
    from .assembler import (assemble_dubbed_audio, format_srt_time,
                            merge_audio_video, write_srt)
    from .audio import (extract_audio, extract_audio_hq, get_duration,
                        separate_background)
    from .diarizer import (assign_speakers_to_segments, diarize_speakers,
                           extract_speaker_audio)
    from .downloader import download_video
    from .models import MODEL_CATALOG, get_system_status
    from .synthesizer import (BaseTTSEngine, CosyVoiceEngine, EdgeTTSFallback,
                              F5TTSEngine, VoxCPMSynthesizer)
    from .transcriber import transcribe
    from .translator import check_ollama, ollama_pull_stream, translate_segments
    from .vad import apply_vad_filter, get_speech_timestamps

__all__ = sorted(_LAZY)
