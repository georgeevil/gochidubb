"""Tests for the progress-aware ffmpeg runner (pipeline/ffmpeg_run.py).

Fake "ffmpeg" processes are python scripts run via sys.executable, so
inject_progress_flags stays out of the way; the runner parses progress
key=value lines from stdout regardless of how they got there.
"""
import sys
import time

import pytest

from pipeline.ffmpeg_run import inject_progress_flags, run_ffmpeg


def _script(tmp_path, body):
    p = tmp_path / "fake_ffmpeg.py"
    p.write_text(body, encoding="utf-8")
    return [sys.executable, "-u", str(p)]


class TestInjectProgressFlags:
    def test_ffmpeg_gets_progress_flags(self):
        cmd = inject_progress_flags(["ffmpeg", "-y", "-i", "in.mp4", "out.mp4"])
        assert cmd[:4] == ["ffmpeg", "-nostats", "-progress", "pipe:1"]
        assert cmd[4:] == ["-y", "-i", "in.mp4", "out.mp4"]

    def test_ffmpeg_with_path_and_exe_suffix(self):
        cmd = inject_progress_flags(["/opt/bin/ffmpeg.exe", "-y", "x"])
        assert cmd[1:4] == ["-nostats", "-progress", "pipe:1"]

    def test_existing_progress_flag_untouched(self):
        orig = ["ffmpeg", "-progress", "out.log", "-i", "in.mp4", "out.mp4"]
        assert inject_progress_flags(orig) == orig

    def test_non_ffmpeg_untouched(self):
        assert inject_progress_flags(["ffprobe", "-v", "error"]) == \
            ["ffprobe", "-v", "error"]


class TestRunFfmpeg:
    def test_success_returns_completed_process(self, tmp_path):
        cmd = _script(tmp_path, "print('hello')\n")
        r = run_ffmpeg(cmd, desc="ok", timeout=10, poll_interval=0.05)
        assert r.returncode == 0
        assert "hello" in r.stdout

    def test_failure_raises_with_stderr(self, tmp_path):
        cmd = _script(
            tmp_path,
            "import sys; sys.stderr.write('boom reason'); sys.exit(1)\n",
        )
        with pytest.raises(RuntimeError, match="boom reason"):
            run_ffmpeg(cmd, desc="bad", timeout=10, poll_interval=0.05)

    def test_hard_timeout_without_progress(self, tmp_path):
        cmd = _script(tmp_path, "import time; time.sleep(30)\n")
        start = time.monotonic()
        with pytest.raises(RuntimeError, match="timed out"):
            run_ffmpeg(cmd, desc="hung", timeout=0.5, stall_timeout=0.5,
                       poll_interval=0.05)
        assert time.monotonic() - start < 10

    def test_progress_extends_past_soft_timeout(self, tmp_path):
        # Runs ~1.5s emitting advancing out_time; soft timeout is 0.5s.
        # Old behaviour would kill it; progress-aware runner lets it finish.
        cmd = _script(tmp_path, (
            "import time\n"
            "for i in range(15):\n"
            "    print(f'out_time=00:00:0{i//10}.{i%10}00000')\n"
            "    print('progress=continue')\n"
            "    time.sleep(0.1)\n"
            "print('progress=end')\n"
        ))
        r = run_ffmpeg(cmd, desc="long render", timeout=0.5, stall_timeout=5,
                       poll_interval=0.05)
        assert r.returncode == 0

    def test_stall_after_progress_is_killed(self, tmp_path):
        cmd = _script(tmp_path, (
            "import time\n"
            "print('out_time=00:00:01.000000')\n"
            "time.sleep(30)\n"
        ))
        start = time.monotonic()
        with pytest.raises(RuntimeError, match="no encode progress"):
            run_ffmpeg(cmd, desc="stalled", timeout=0.3, stall_timeout=0.5,
                       poll_interval=0.05)
        assert time.monotonic() - start < 10

    def test_repeated_identical_progress_is_a_stall(self, tmp_path):
        # ffmpeg can keep printing progress blocks with a frozen position;
        # unchanged values must not count as progress.
        cmd = _script(tmp_path, (
            "import time\n"
            "for _ in range(100):\n"
            "    print('out_time=00:00:01.000000')\n"
            "    time.sleep(0.1)\n"
        ))
        start = time.monotonic()
        with pytest.raises(RuntimeError, match="timed out"):
            run_ffmpeg(cmd, desc="frozen", timeout=0.3, stall_timeout=0.5,
                       poll_interval=0.05)
        assert time.monotonic() - start < 10

    def test_progress_lines_stripped_from_stdout(self, tmp_path):
        cmd = _script(tmp_path, (
            "print('out_time=00:00:01.000000')\n"
            "print('speed=1.5x')\n"
            "print('real output line')\n"
        ))
        r = run_ffmpeg(cmd, desc="mixed", timeout=10, poll_interval=0.05)
        assert "real output line" in r.stdout
        assert "out_time" not in r.stdout
        assert "speed" not in r.stdout
