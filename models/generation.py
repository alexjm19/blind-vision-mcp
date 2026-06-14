import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image

from mcp_server.config import settings


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GenerationModel:
    def __init__(self):
        self.model_id = settings.generation_model_id
        self._pipe = None
        self._device = None
        self._load()

    def _load(self):
        from diffusers import FluxPipeline

        logger.info(f"Loading generation model: {self.model_id}")

        dtype = torch.bfloat16
        if settings.generation_fp8:
            dtype = torch.float8_e4m3fn

        if torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        self._pipe = FluxPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto" if self._device == "cuda" else None,
        )

        if self._device == "cuda":
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe = self._pipe.to(self._device)

        self._pipe.eval()
        logger.info(f"Generation model {self.model_id} loaded on {self._device}")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        output_path: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 4,
        guidance_scale: float = 0.0,
        seed: int | None = None,
        negative_prompt: str | None = None,
    ) -> dict:
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        _seed_everything(seed)

        generator = torch.Generator(device=self._device).manual_seed(seed)

        kwargs = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "output_type": "pil",
        }
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt

        image = self._pipe(**kwargs).images[0]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)

        return {"seed": seed}


class EditModel:
    def __init__(self):
        self.model_id = settings.edit_model_id
        self._pipe = None
        self._device = None
        self._load()

    def _load(self):
        from diffusers import FluxFillPipeline

        logger.info(f"Loading edit model: {self.model_id}")

        dtype = torch.bfloat16
        if settings.edit_fp8:
            dtype = torch.float8_e4m3fn

        if torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        self._pipe = FluxFillPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map="auto" if self._device == "cuda" else None,
        )

        if self._device == "cuda":
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe = self._pipe.to(self._device)

        self._pipe.eval()
        logger.info(f"Edit model {self.model_id} loaded on {self._device}")

    @torch.no_grad()
    def edit(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        output_path: str,
        steps: int = 50,
        strength: float = 0.85,
        seed: int | None = None,
    ) -> dict:
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        _seed_everything(seed)

        generator = torch.Generator(device=self._device).manual_seed(seed)

        result = self._pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=steps,
            strength=strength,
            generator=generator,
            output_type="pil",
        ).images[0]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)

        return {"seed": seed}
