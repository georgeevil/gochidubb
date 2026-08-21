"""Tests for pipeline/models.py — model size estimation and system checks."""

import pytest

from pipeline.models import _estimate_model_size, MODEL_CATALOG


class TestEstimateModelSize:
    """_estimate_model_size() is a pure function mapping model IDs to GB."""

    def test_qwen_72b(self):
        assert _estimate_model_size("Qwen/Qwen-72B-Chat") == 40.0
        assert _estimate_model_size("qwen-72b") == 40.0

    def test_qwen_32b(self):
        assert _estimate_model_size("Qwen/Qwen-32B") == 18.0

    def test_qwen_14b(self):
        assert _estimate_model_size("Qwen/Qwen-14B") == 8.0

    def test_qwen_7b(self):
        assert _estimate_model_size("Qwen/Qwen-7B") == 4.0

    def test_qwen_3b(self):
        assert _estimate_model_size("Qwen/Qwen-3B") == 2.0

    def test_qwen_1_5b(self):
        assert _estimate_model_size("Qwen/Qwen-1.5B") == 1.0

    def test_gemma_27b(self):
        assert _estimate_model_size("google/gemma-2-27b-it") == 16.0

    def test_gemma_12b(self):
        assert _estimate_model_size("google/gemma-2-12b-it") == 7.5

    def test_gemma_4b(self):
        assert _estimate_model_size("google/gemma-2-4b-it") == 3.0

    def test_gemma_2b(self):
        assert _estimate_model_size("google/gemma-2b") == 1.5

    def test_llama_70b(self):
        assert _estimate_model_size("meta-llama/Llama-2-70b") == 40.0

    def test_llama_8b(self):
        assert _estimate_model_size("meta-llama/Llama-3-8b") == 4.5

    def test_llama_3b(self):
        assert _estimate_model_size("meta-llama/Llama-3.2-3B") == 2.0

    def test_mistral_7b(self):
        assert _estimate_model_size("mistralai/Mistral-7B") == 4.0

    def test_mixtral_8x7b(self):
        # "mistralai/Mixtral-8x7B" → "mistral" IN model_lower → mistral branch
        # "7b" in model_lower → returns 4.0
        assert _estimate_model_size("mistralai/Mixtral-8x7B") == 4.0

    def test_phi_3_mini(self):
        # "phi" not in any known family; no "Xb" pattern → 0.0
        assert _estimate_model_size("microsoft/Phi-3-mini-4k-instruct") == 0.0

    def test_phi_2(self):
        # "phi" not in any known family; "2" has no "b" suffix → 0.0
        assert _estimate_model_size("microsoft/phi-2") == 0.0

    def test_deepseek_v2(self):
        # "deepseek" not in any known family; "v2.5" has no "b" suffix → 0.0
        assert _estimate_model_size("deepseek-ai/deepseek-v2.5") == 0.0

    def test_deepseek_coder_v2(self):
        # "deepseek" not in any known family; no "Xb" pattern → 0.0
        assert _estimate_model_size("deepseek-ai/deepseek-coder-v2") == 0.0

    def test_deepseek_r1_distill(self):
        # "deepseek" not in known families, but "Qwen-7B" matched by Qwen rules → 4.0
        assert _estimate_model_size("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B") == 4.0

    def test_deepseek_coder_6b(self):
        # Falls through to general fallback: re.search(r'(\d+)b', "6.7b") → "7b" → 7*0.6 = 4.2
        result = _estimate_model_size("deepseek-ai/deepseek-coder-6.7b")
        assert result == pytest.approx(4.2, abs=0.1)

    def test_unrecognized_model_returns_zero(self):
        assert _estimate_model_size("some-unknown-model") == 0.0

    def test_case_insensitive_matching(self):
        assert _estimate_model_size("QWEN/QWEN-72B-CHAT") == 40.0
        assert _estimate_model_size("Meta-Llama/Llama-3-8B") == 4.5

    def test_deepseek_r1_70b(self):
        # Not in known families; fallback: re.search matches "70b" → 70 * 0.6 = 42.0
        assert _estimate_model_size("deepseek-ai/DeepSeek-R1-70B") == 42.0

    def test_phi_medium(self):
        # "phi" not in known families; no "Xb" pattern → 0.0
        assert _estimate_model_size("microsoft/Phi-3-medium") == 0.0

    def test_phi_small(self):
        # "phi" not in known families; no "Xb" pattern → 0.0
        assert _estimate_model_size("microsoft/Phi-3-small") == 0.0

    def test_codeqwen(self):
        assert _estimate_model_size("Qwen/CodeQwen1.5-7B") == 4.0

    def test_nvidia_nemotron(self):
        assert _estimate_model_size("nvidia/Llama-3.1-Nemotron-70B") == 40.0

    def test_empty_string(self):
        assert _estimate_model_size("") == 0.0


class TestModelCatalog:
    """MODEL_CATALOG is a dict with an empty 'translation' list by default."""

    def test_catalog_is_dict(self):
        assert isinstance(MODEL_CATALOG, dict)

    def test_has_translation_key(self):
        assert "translation" in MODEL_CATALOG

    def test_translation_is_list(self):
        assert isinstance(MODEL_CATALOG["translation"], list)
