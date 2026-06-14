
import pytest
from PIL import Image

from utils.image_utils import is_url, load_image


class TestIsURL:
    def test_http_url(self):
        assert is_url("http://example.com/image.jpg") is True

    def test_https_url(self):
        assert is_url("https://example.com/image.png") is True

    def test_local_path(self):
        assert is_url("/home/user/image.png") is False

    def test_relative_path(self):
        assert is_url("images/test.png") is False


class TestLoadImage:
    def test_load_local_image(self, tmp_path):
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (50, 50), color="red")
        img.save(img_path)

        loaded = load_image(str(img_path))
        assert loaded.size == (50, 50)
        assert loaded.mode == "RGB"

    def test_load_image_converts_grayscale(self, tmp_path):
        img_path = tmp_path / "gray.png"
        img = Image.new("L", (30, 30), color=128)
        img.save(img_path)

        loaded = load_image(str(img_path))
        assert loaded.mode == "RGB"


class TestDescribe:
    @pytest.mark.asyncio
    async def test_describe_basic(self, mock_vision_model, test_image):
        result = mock_vision_model.describe(test_image, "Describe this image")
        assert "DESCRIPCIÓN GENERAL" in result
        assert "COMPOSICIÓN" in result

    @pytest.mark.asyncio
    async def test_describe_with_focus(self, mock_vision_model, test_image):
        mock_vision_model.describe(
            test_image, "Describe this image. Focus specifically on: colors."
        )
        assert mock_vision_model.describe.called


class TestCompare:
    @pytest.mark.asyncio
    async def test_compare_two_images(self, mock_vision_model, test_image):
        result = mock_vision_model.compare(test_image, test_image, "Compare these images")
        assert "SIMILITUDES" in result
        assert "DIFERENCIAS PRINCIPALES" in result
        assert "CONCLUSIÓN" in result

    @pytest.mark.asyncio
    async def test_compare_with_focus(self, mock_vision_model, test_image):
        mock_vision_model.compare(
            test_image, test_image, "Compare these images. Focus specifically on: color."
        )
        assert mock_vision_model.compare.called
