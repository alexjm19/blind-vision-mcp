import pytest

from utils.prompt_optimizer import build_edit_prompt, build_generation_prompt


class TestPromptOptimizer:
    def test_build_generation_prompt_adds_quality_tokens(self):
        result = build_generation_prompt("a cat sitting on a chair")
        assert "masterpiece" in result
        assert "best quality" in result
        assert "A stunning, high-quality photograph of" in result
        assert "a cat sitting on a chair" in result

    def test_build_generation_prompt_trims_trailing_period(self):
        result = build_generation_prompt("a cat sitting on a chair.")
        assert not result.endswith("..") or result.count(".") <= 2

    def test_build_generation_prompt_long_input(self):
        long_desc = "a " * 500
        result = build_generation_prompt(long_desc)
        assert len(result) <= 1024

    def test_build_edit_prompt(self):
        result = build_edit_prompt("a red car")
        assert "Transform the masked region to:" in result
        assert "a red car" in result


class TestGeneration:
    @pytest.mark.asyncio
    async def test_generate_returns_seed(self, mock_generation_model):
        result = mock_generation_model.generate(
            prompt="test prompt",
            output_path="/tmp/test.png",
        )
        assert "seed" in result

    @pytest.mark.asyncio
    async def test_edit_returns_seed(self, mock_edit_model, test_image, test_mask):
        result = mock_edit_model.edit(
            image=test_image,
            mask=test_mask,
            prompt="add a tree",
            output_path="/tmp/edit.png",
        )
        assert "seed" in result
