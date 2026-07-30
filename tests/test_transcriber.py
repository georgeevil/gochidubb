"""Tests for pipeline/transcriber.py — thread count logic and edge cases."""
from unittest.mock import patch

from pipeline.transcriber import get_optimal_thread_count, transcribe_fallback


class TestGetOptimalThreadCount:
    """get_optimal_thread_count() returns appropriate thread counts per CPU size."""

    @patch("multiprocessing.cpu_count", return_value=2)
    def test_small_system(self, mock_cpu):
        """<=4 cores: use all but 1."""
        assert get_optimal_thread_count() == 1

    @patch("multiprocessing.cpu_count", return_value=4)
    def test_four_cores(self, mock_cpu):
        """4 cores → 3 threads."""
        assert get_optimal_thread_count() == 3

    @patch("multiprocessing.cpu_count", return_value=6)
    def test_six_cores(self, mock_cpu):
        """6 cores (Apple M1 base): use 4 threads."""
        assert get_optimal_thread_count() == 4

    @patch("multiprocessing.cpu_count", return_value=8)
    def test_eight_cores(self, mock_cpu):
        """8 cores: use 6 threads."""
        assert get_optimal_thread_count() == 6

    @patch("multiprocessing.cpu_count", return_value=10)
    def test_ten_cores(self, mock_cpu):
        """10 cores (Apple M1 Pro/Max): use 75% → 7."""
        assert get_optimal_thread_count() == 7

    @patch("multiprocessing.cpu_count", return_value=16)
    def test_sixteen_cores(self, mock_cpu):
        """16 cores: use 75% → 12."""
        assert get_optimal_thread_count() == 12

    @patch("multiprocessing.cpu_count", return_value=64)
    def test_epyc_system(self, mock_cpu):
        """64 cores: use 75% → 48."""
        assert get_optimal_thread_count() == 48

    @patch("multiprocessing.cpu_count", return_value=1)
    def test_single_core(self, mock_cpu):
        """Single core: at least 1."""
        assert get_optimal_thread_count() == 1
