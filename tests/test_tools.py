import json
from unittest.mock import patch

import pytest
from PIL import Image

from mcp_server.tools import (
    get_status,
    image_edit,
    image_generate,
    vision_compare,
    vision_describe,
)


def _make_patch_path():
    return patch("mcp_server.tools.load_image", return_value=Image.new("RGB", (100, 100)))


@pytest.fixture
def mock_manager(mock_model_manager):
    return mock_model_manager


@pytest.mark.asyncio
async def test_describe_tool_calls_model(mock_manager):
    with patch("mcp_server.tools.model_manager", mock_manager), _make_patch_path():
        result = await vision_describe("/fake/path.png")
        assert "DESCRIPCIÓN GENERAL" in result
        mock_manager.get_vision_model.assert_called_once()


@pytest.mark.asyncio
async def test_describe_tool_error_returns_error_message(mock_manager):
    mock_manager.get_vision_model.side_effect = RuntimeError("test error")
    with patch("mcp_server.tools.model_manager", mock_manager), _make_patch_path():
        result = await vision_describe("/fake/path.png")
        assert "Error" in result


@pytest.mark.asyncio
async def test_generate_tool_returns_json(mock_manager):
    with patch("mcp_server.tools.model_manager", mock_manager):
        result = await image_generate("a test image")
        data = json.loads(result)
        assert "image_path" in data
        assert "prompt_used" in data
        assert "seed" in data


@pytest.mark.asyncio
async def test_edit_tool_returns_json(mock_manager, temp_output):
    with patch("mcp_server.tools.model_manager", mock_manager), patch(
        "mcp_server.tools.load_image", return_value=Image.new("RGB", (100, 100))
    ):
        result = await image_edit("/fake/image.png", "/fake/mask.png", "add a sun")
        data = json.loads(result)
        assert "image_path" in data
        assert "prompt_used" in data


@pytest.mark.asyncio
async def test_compare_tool_returns_text(mock_manager):
    with patch("mcp_server.tools.model_manager", mock_manager), patch(
        "mcp_server.tools.load_image", return_value=Image.new("RGB", (100, 100))
    ):
        result = await vision_compare("/fake/a.png", "/fake/b.png")
        assert "SIMILITUDES" in result


@pytest.mark.asyncio
async def test_status_tool_returns_json(mock_manager):
    with patch("mcp_server.tools.model_manager", mock_manager):
        result = await get_status()
        data = json.loads(result)
        assert "loaded_model" in data
        assert "vram" in data
