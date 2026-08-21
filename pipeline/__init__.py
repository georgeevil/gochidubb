"""GoChiDUBB Studio Pipeline - modular dubbing components."""
from .downloader import download_video
from .audio import extract_audio, extract_audio_hq, separate_background, get_duration
from .transcriber import transcribe
from .diarizer import diarize_speakers, assign_speakers_to_segments, extract_speaker_audio
from .translator import translate_segments, check_ollama, ollama_pull_stream
from .synthesizer import BaseTTSEngine, VoxCPMSynthesizer, F5TTSEngine, CosyVoiceEngine, EdgeTTSFallback
from .assembler import assemble_dubbed_audio, merge_audio_video, write_srt, format_srt_time
from .models import get_system_status, MODEL_CATALOG
from .vad import apply_vad_filter, get_speech_timestamps

# Re-export module: everything above is imported for callers, not for use
# here. __all__ states that explicitly, which is also what stops a linter
# reading the imports as dead code.
__all__ = [
    "download_video",
    "extract_audio", "extract_audio_hq", "separate_background", "get_duration",
    "transcribe",
    "diarize_speakers", "assign_speakers_to_segments", "extract_speaker_audio",
    "translate_segments", "check_ollama", "ollama_pull_stream",
    "BaseTTSEngine", "VoxCPMSynthesizer", "F5TTSEngine", "CosyVoiceEngine",
    "EdgeTTSFallback",
    "assemble_dubbed_audio", "merge_audio_video", "write_srt", "format_srt_time",
    "get_system_status", "MODEL_CATALOG",
    "apply_vad_filter", "get_speech_timestamps",
]
