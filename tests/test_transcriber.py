"""Tests for pipeline/transcriber.py — thread count logic and edge cases."""
from unittest.mock import patch

from pipeline.transcriber import (
    get_optimal_thread_count,
    word_confidence_stats,
)


class TestWordConfidenceStats:
    """word_confidence_stats() aggregates per-word ASR confidences."""

    def test_mean_and_min(self):
        words = [
            {"word": "a", "probability": 0.9},
            {"word": "b", "probability": 0.5},
            {"word": "c", "probability": 0.7},
        ]
        stats = word_confidence_stats(words)
        assert stats["word_conf_mean"] == 0.7
        assert stats["word_conf_min"] == 0.5

    def test_empty_words_returns_empty(self):
        assert word_confidence_stats([]) == {}
        assert word_confidence_stats(None) == {}

    def test_words_without_probability_returns_empty(self):
        """Never fabricate a confidence that was not reported."""
        assert word_confidence_stats([{"word": "a"}, {"word": "b"}]) == {}

    def test_ignores_non_dict_entries_and_bad_values(self):
        words = [
            "not-a-dict",
            {"word": "a", "probability": "high"},
            {"word": "b", "probability": True},   # bool is not a confidence
            {"word": "c", "probability": 0.8},
        ]
        stats = word_confidence_stats(words)
        assert stats == {"word_conf_mean": 0.8, "word_conf_min": 0.8}

    def test_rounds_to_four_places(self):
        words = [{"probability": 1 / 3}, {"probability": 2 / 3}]
        stats = word_confidence_stats(words)
        assert stats["word_conf_mean"] == 0.5
        assert stats["word_conf_min"] == round(1 / 3, 4)


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
