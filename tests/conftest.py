import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def test_image():
    img = Image.new("RGB", (100, 100), color=(73, 109, 137))
    return img


@pytest.fixture
def test_mask():
    mask = Image.new("L", (100, 100), color=0)
    mask.paste(255, (25, 25, 75, 75))
    return mask


@pytest.fixture
def mock_vision_model():
    model = MagicMock()
    model.describe.return_value = (
        "DESCRIPCIÓN GENERAL\nA blue square on a white background.\n"
        "COMPOSICIÓN\nCentered composition.\n"
        "COLORES E ILUMINACIÓN\nBlue and white."
    )
    model.compare.return_value = (
        "SIMILITUDES\nBoth are squares.\n"
        "DIFERENCIAS PRINCIPALES\nFirst is blue, second is red.\n"
        "CONCLUSIÓN\nColor difference."
    )
    return model


@pytest.fixture
def mock_generation_model():
    model = MagicMock()
    model.generate.return_value = {"seed": 42}
    model._pipe = MagicMock()
    return model


@pytest.fixture
def mock_edit_model():
    model = MagicMock()
    model.edit.return_value = {"seed": 99}
    model._pipe = MagicMock()
    return model


@pytest.fixture
def mock_model_manager(mock_vision_model, mock_generation_model, mock_edit_model):
    manager = MagicMock()
    manager.get_vision_model = AsyncMock(return_value=mock_vision_model)
    manager.get_generation_model = AsyncMock(return_value=mock_generation_model)
    manager.get_edit_model = AsyncMock(return_value=mock_edit_model)
    manager._loaded_model = "vision"
    manager.get_vram_stats.return_value = {
        "cuda_available": False,
        "loaded_model": "vision",
    }
    return manager


@pytest.fixture
def temp_output(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir
