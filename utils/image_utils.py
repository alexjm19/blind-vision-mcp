import base64
import tempfile
from pathlib import Path

import cv2
import numpy as np
import requests
from loguru import logger
from PIL import Image


def is_url(path: str) -> bool:
    return path.startswith(("http://", "https://"))


def is_data_url(path: str) -> bool:
    return path.startswith("data:image/")


def load_data_url(path: str) -> Image.Image:
    header, _, b64_data = path.partition(",")
    image_bytes = base64.b64decode(b64_data)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(image_bytes)
    tmp.close()
    return Image.open(tmp.name)


def load_image(path: str) -> Image.Image:
    if is_url(path):
        logger.info(f"Downloading image from URL: {path}")
        resp = requests.get(path, timeout=60, stream=True)
        resp.raise_for_status()
        image = Image.open(resp.raw)
    elif is_data_url(path):
        logger.info("Decoding image from data URL")
        image = load_data_url(path)
    else:
        image = Image.open(Path(path))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def feather_mask(mask: Image.Image, feather_px: int = 5) -> Image.Image:
    mask_np = np.array(mask.convert("L"))

    if feather_px > 0:
        kernel_size = feather_px * 2 + 1
        mask_np = cv2.GaussianBlur(mask_np, (kernel_size, kernel_size), 0)

    mask_np = np.clip(mask_np, 0, 255).astype(np.uint8)
    return Image.fromarray(mask_np, mode="L")


def resize_for_model(image: Image.Image, max_size: int = 1024) -> Image.Image:
    w, h = image.size
    if max(w, h) <= max_size:
        return image

    if w > h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)

    return image.resize((new_w, new_h), Image.LANCZOS)


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)
