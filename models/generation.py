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


def _get_pipeline_cls(model_id: str):
    from diffusers import (
        AutoPipelineForText2Image,
        Flux2KleinPipeline,
        FluxPipeline,
    )

    if "klein" in model_id.lower() or "flux2" in model_id.lower():
        return Flux2KleinPipeline
    if "flux" in model_id.lower():
        return FluxPipeline
    return AutoPipelineForText2Image


class GenerationModel:
    def __init__(self):
        self.model_id = settings.generation_model_id
        self._pipe = None
        self._device = None
        self._load()

    def _load(self):
        logger.info(f"Loading generation model: {self.model_id}")

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        cls = _get_pipeline_cls(self.model_id)
        hf_token = __import__("os").environ.get("HF_TOKEN") or None

        dtype = torch.float16 if "sdxl" in self.model_id.lower() else torch.bfloat16
        kwargs = {"torch_dtype": dtype, "token": hf_token}
        if dtype == torch.float16:
            kwargs["variant"] = "fp16"

        self._pipe = cls.from_pretrained(self.model_id, **kwargs)

        if self._device == "cuda" and "turbo" not in self.model_id.lower():
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe = self._pipe.to(self._device)

        if hasattr(self._pipe, "eval"):
            self._pipe.eval()
        logger.info(f"Generation model loaded on {self._device}")

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

        generator = torch.Generator(device="cuda").manual_seed(seed)
        result = self._pipe(
            prompt=prompt,
            generator=generator,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
        ).images[0]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
        return {"seed": seed, "image_path": output_path}


class EditModel:
    def __init__(self):
        self.model_id = settings.edit_model_id
        self._pipe = None
        self._device = None
        self._load()

    def _load(self):
        from diffusers import FluxInpaintPipeline

        logger.info(f"Loading edit model: {self.model_id}")
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        hf_token = __import__("os").environ.get("HF_TOKEN") or None

        self._pipe = FluxInpaintPipeline.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, token=hf_token
        )

        if self._device == "cuda":
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe = self._pipe.to(self._device)

        if hasattr(self._pipe, "eval"):
            self._pipe.eval()
        logger.info(f"Edit model loaded on {self._device}")

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
        width, height = image.size
        if width > 1024 or height > 1024:
            scale = 1024 / max(width, height)
            image = image.resize(
                (int(width * scale), int(height * scale)), Image.LANCZOS
            )
            mask = mask.resize(
                (int(width * scale), int(height * scale)), Image.LANCZOS
            )

        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        _seed_everything(seed)

        generator = torch.Generator(device="cuda").manual_seed(seed)
        result = self._pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            generator=generator,
            num_inference_steps=steps,
            strength=strength,
        ).images[0]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
        return {"seed": seed, "image_path": output_path}
