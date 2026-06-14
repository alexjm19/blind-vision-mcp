import base64
import tempfile
import time
from pathlib import Path
from typing import Union

import requests
from loguru import logger
from PIL import Image


LITERT_PORT = 9380
LITERT_MODEL_REF = "gemma4-vision"


class LiteRTVisionModel:
    def __init__(self):
        self._base_url = f"http://localhost:{LITERT_PORT}"

    def _ensure_server(self):
        for attempt in range(30):
            try:
                resp = requests.get(f"{self._base_url}/v1/models", timeout=2)
                if resp.status_code == 200:
                    return
            except requests.ConnectionError:
                pass
            time.sleep(1)
        raise RuntimeError("LiteRT server not ready after 30s")

    def _img_to_b64(self, img: Union[str, Image.Image]) -> tuple[str, str]:
        if isinstance(img, str):
            pil = Image.open(Path(img))
        else:
            pil = img
        # Resize to speed up vision encoding (768px max)
        w, h = pil.size
        if max(w, h) > 768:
            scale = 768 / max(w, h)
            pil = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        pil.save(tmp, format="PNG")
        tmp.close()
        with open(tmp.name, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return "png", b64

    def _infer(self, images: list[Union[str, Image.Image]], prompt: str) -> str:
        self._ensure_server()
        content = []
        for img in images:
            ext, b64 = self._img_to_b64(img)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{ext};base64,{b64}"},
                }
            )
        content.append({"type": "text", "text": prompt})

        for model_id in [f"{LITERT_MODEL_REF},gpu", LITERT_MODEL_REF]:
            try:
                resp = requests.post(
                    f"{self._base_url}/v1/chat/completions",
                    json={
                        "model": model_id,
                        "messages": [{"role": "user", "content": content}],
                        "max_tokens": 4096,
                    },
                    timeout=180,
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
                logger.warning(f"Model {model_id} failed: {resp.status_code}")
            except Exception as e:
                logger.warning(f"Model {model_id} error: {e}")

        raise RuntimeError("All LiteRT models failed")

    def describe(self, image: Union[str, Image.Image], prompt: str) -> str:
        return self._infer([image], prompt)

    def compare(
        self,
        image_a: Union[str, Image.Image],
        image_b: Union[str, Image.Image],
        prompt: str,
    ) -> str:
        return self._infer([image_a, image_b], prompt)

    def unload(self):
        pass
