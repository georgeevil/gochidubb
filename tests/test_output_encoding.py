"""Tests for output encoding decisions (pipeline/assembler.py).

The rule these pin down: stream-copy the video only when it is already
playable, and re-encode when it is not. Copying blindly is what put a VP9
stream inside an .mp4 — a structurally valid file that QuickTime, WhatsApp
and Finder all refuse, while YouTube plays it because YouTube re-encodes on
upload. Always re-encoding would be safe but costs minutes of GPU time per
job for sources that were already fine.
"""
from unittest.mock import patch

import pytest

from pipeline.assembler import _video_codec_args


def _args(codec, pix_fmt, must_encode=False, pref="h264", preset="veryfast", crf=23):
    with patch("pipeline.assembler._probe_video_stream",
               return_value=(codec, pix_fmt)):
        return _video_codec_args("v.mp4", must_encode, pref, preset, crf)


def _is_copy(args):
    return args[:2] == ["-c:v", "copy"]


class TestCopyWhenSafe:
    def test_h264_yuv420p_is_copied(self):
        # The fast path: already what every player wants, so don't transcode.
        assert _is_copy(_args("h264", "yuv420p"))

    def test_jpeg_range_420_is_also_copied(self):
        assert _is_copy(_args("h264", "yuvj420p"))


class TestReencodeWhenNot:
    @pytest.mark.parametrize("codec,pix_fmt", [
        ("vp9", "yuv420p"),      # the reported bug
        ("av01", "yuv420p"),
        ("hevc", "yuv420p"),
        ("h264", "yuv444p"),     # right codec, wrong chroma — QT still refuses
        ("h264", "yuv420p10le"), # 10-bit
    ])
    def test_incompatible_source_is_reencoded(self, codec, pix_fmt):
        args = _args(codec, pix_fmt)
        assert not _is_copy(args)
        assert "libx264" in args
        assert args[args.index("-pix_fmt") + 1] == "yuv420p"

    def test_unprobeable_source_is_reencoded(self):
        # No ffprobe, or an unreadable file: re-encode rather than gamble.
        assert not _is_copy(_args(None, None))

    def test_filtered_output_must_encode_even_if_source_is_safe(self):
        # The frame-hold path applies a filter, so copying is impossible.
        assert not _is_copy(_args("h264", "yuv420p", must_encode=True))


class TestUserPreference:
    def test_copy_preference_skips_the_probe_entirely(self):
        # An explicit "copy" is the user choosing speed over reach.
        with patch("pipeline.assembler._probe_video_stream") as probe:
            args = _video_codec_args("v.mp4", False, "copy", "veryfast", 23)
        assert _is_copy(args)
        probe.assert_not_called()

    def test_copy_preference_still_encodes_when_a_filter_is_applied(self):
        assert not _is_copy(_args("vp9", "yuv420p", must_encode=True, pref="copy"))


class TestEncodeSettingsArePassedThrough:
    def test_preset_and_crf_reach_ffmpeg(self):
        args = _args("vp9", "yuv420p", preset="slow", crf=18)
        assert args[args.index("-preset") + 1] == "slow"
        assert args[args.index("-crf") + 1] == "18"

    def test_crf_is_stringified(self):
        # ffmpeg args must all be strings; an int here raises at subprocess time.
        assert all(isinstance(a, str) for a in _args("vp9", "yuv420p", crf=20))
