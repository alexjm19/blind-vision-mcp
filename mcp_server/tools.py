import json
import time
from pathlib import Path
from typing import Optional

from loguru import logger
from mcp.server.fastmcp import Context

from mcp_server.config import settings
from models.manager import model_manager
from utils.image_utils import feather_mask, load_image
from utils.prompt_optimizer import build_generation_prompt


DETAIL_PROMPTS = {
    "brief": "Describe this image concisely in 2-3 sentences.",
    "standard": "Describe this image in detail.",
    "ultra": (
        "Analyze this image in extreme detail. Structure your response with these sections:\n"
        "DESCRIPCIÓN GENERAL\n"
        "COMPOSICIÓN\n"
        "SUJETOS PRINCIPALES\n"
        "COLORES E ILUMINACIÓN\n"
        "TEXTURAS Y MATERIALES\n"
        "TEXTO VISIBLE\n"
        "ANOMALÍAS\n"
        "CONTEXTO\n"
        "DETALLES TÉCNICOS\n"
        "Be thorough and specific in every section."
    ),
}

COMPARE_PROMPT = (
    "Compare these two images in detail. Structure your response with:\n"
    "SIMILITUDES\n"
    "DIFERENCIAS PRINCIPALES\n"
    "DIFERENCIAS EN DETALLE\n"
    "CONCLUSIÓN\n"
    "Be specific about what changed between image A and image B."
)


async def vision_describe(
    ctx: Context,
    image: str,
    detail_level: str = "ultra",
    focus: Optional[str] = None,
) -> str:
    """Analyze an image and return a detailed text description.

    Supports file paths, HTTP URLs, and data:image/...;base64,... data URLs.

    Args:
        image: Image path, URL, or data:image/...;base64,... data URL.
        detail_level: Level of detail: 'brief', 'standard', or 'ultra'.
        focus: Optional focus area (e.g., 'defects', 'text', 'people').
    """
    logger.info(f"vision_describe called: image={image[:80]}..., detail={detail_level}, focus={focus}")
    try:
        await ctx.info("Loading image...")
        img = load_image(image)
        model = await model_manager.get_vision_model(progress_cb=lambda msg: ctx.info(msg))
        await ctx.info("Analyzing image with vision model...")
        prompt = DETAIL_PROMPTS.get(detail_level, DETAIL_PROMPTS["ultra"])
        if focus:
            prompt += f"\n\nFocus specifically on: {focus}."
        result = model.describe(img, prompt)
        logger.info(f"vision_describe completed ({len(result)} chars)")
        return result
    except Exception as e:
        logger.error(f"vision_describe failed: {e}")
        return f"Error analyzing image: {e}"


async def vision_compare(
    ctx: Context,
    image_a: str,
    image_b: str,
    focus: Optional[str] = None,
) -> str:
    """Compare two images and describe their differences.

    Supports file paths, HTTP URLs, and data:image/...;base64,... data URLs.

    Args:
        image_a: First image path, URL, or data URL.
        image_b: Second image path, URL, or data URL.
        focus: Optional focus area for comparison.
    """
    logger.info(f"vision_compare called: {image_a[:60]} vs {image_b[:60]}, focus={focus}")
    try:
        await ctx.info("Loading images...")
        img_a = load_image(image_a)
        img_b = load_image(image_b)
        model = await model_manager.get_vision_model(progress_cb=lambda msg: ctx.info(msg))
        await ctx.info("Comparing images...")
        prompt = COMPARE_PROMPT
        if focus:
            prompt += f"\nFocus specifically on: {focus}."
        result = model.compare(img_a, img_b, prompt)
        logger.info(f"vision_compare completed ({len(result)} chars)")
        return result
    except Exception as e:
        logger.error(f"vision_compare failed: {e}")
        return f"Error comparing images: {e}"


async def image_generate(
    ctx: Context,
    description: str,
    output_path: Optional[str] = None,
    params: Optional[dict] = None,
) -> str:
    """Generate an image from a text description.

    Args:
        description: Natural language description of the image to generate.
        output_path: Optional path to save the image.
        params: Optional dict with: width, height, steps, guidance_scale, seed,
                negative_prompt, override_prompt.
    """
    logger.info(f"image_generate called: '{description[:80]}...' params={params}")
    params = params or {}
    try:
        model = await model_manager.get_generation_model(progress_cb=lambda msg: ctx.info(msg))
        width = params.get("width", 1024)
        height = params.get("height", 1024)
        steps = params.get("steps", 4)
        guidance_scale = params.get("guidance_scale", 0.0)
        seed = params.get("seed", None)
        negative_prompt = params.get("negative_prompt", None)
        override_prompt = params.get("override_prompt", False)

        if override_prompt:
            prompt = description
        else:
            prompt = build_generation_prompt(description)

        if output_path is None:
            timestamp = int(time.time())
            Path(settings.default_output_dir).mkdir(parents=True, exist_ok=True)
            output_path = str(Path(settings.default_output_dir) / f"generated_{timestamp}.png")

        await ctx.info(f"Generating image: {width}x{height}, {steps} steps...")
        start = time.time()
        result = model.generate(
            prompt=prompt,
            output_path=output_path,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            negative_prompt=negative_prompt,
        )
        elapsed = round(time.time() - start, 2)

        response = {
            "image_path": output_path,
            "prompt_used": prompt,
            "seed": result.get("seed", seed),
            "model": settings.generation_model_id,
            "resolution": f"{width}x{height}",
            "generation_time_seconds": elapsed,
        }
        await ctx.info(f"Image generated in {elapsed}s: {output_path}")
        logger.info(f"image_generate completed: {output_path} ({elapsed}s)")
        return json.dumps(response, indent=2)
    except Exception as e:
        logger.error(f"image_generate failed: {e}")
        return json.dumps({"error": str(e)})


async def image_edit(
    ctx: Context,
    image: str,
    mask: str,
    description: str,
    output_path: Optional[str] = None,
    params: Optional[dict] = None,
) -> str:
    """Edit a zone of an image defined by a binary mask.

    Supports file paths, HTTP URLs, and data:image/...;base64,... data URLs.

    Args:
        image: Image path, URL, or data URL.
        mask: Binary mask PNG path, URL, or data URL (white = editable zone).
        description: Description of what to generate in the masked zone.
        output_path: Optional path to save result.
        params: Optional dict with: seed, steps, strength, feather_px.
    """
    short_desc = description[:60]
    logger.info(f"image_edit called: {image[:40]}..., mask={mask[:40]}..., desc='{short_desc}...'")
    params = params or {}
    try:
        await ctx.info("Loading images...")
        img = load_image(image)
        msk = load_image(mask)
        feather_px = params.get("feather_px", 5)
        if feather_px > 0:
            msk = feather_mask(msk, feather_px)

        model = await model_manager.get_edit_model(progress_cb=lambda msg: ctx.info(msg))
        steps = params.get("steps", 50)
        strength = params.get("strength", 0.85)
        seed = params.get("seed", None)

        if output_path is None:
            timestamp = int(time.time())
            Path(settings.default_output_dir).mkdir(parents=True, exist_ok=True)
            output_path = str(Path(settings.default_output_dir) / f"edited_{timestamp}.png")

        await ctx.info(f"Editing image: {steps} steps, strength={strength}...")
        start = time.time()
        result = model.edit(
            image=img,
            mask=msk,
            prompt=description,
            output_path=output_path,
            steps=steps,
            strength=strength,
            seed=seed,
        )
        elapsed = round(time.time() - start, 2)

        response = {
            "image_path": output_path,
            "prompt_used": description,
            "seed": result.get("seed", seed),
            "model": settings.edit_model_id,
            "edit_time_seconds": elapsed,
        }
        await ctx.info(f"Image edited in {elapsed}s: {output_path}")
        logger.info(f"image_edit completed: {output_path} ({elapsed}s)")
        return json.dumps(response, indent=2)
    except Exception as e:
        logger.error(f"image_edit failed: {e}")
        return json.dumps({"error": str(e)})


async def get_status(ctx: Context) -> str:
    """Get server status: loaded models, VRAM usage, queue state."""
    try:
        vram = model_manager.get_vram_stats()
        status = {
            "loaded_model": model_manager._loaded_model,
            "vram": vram,
            "queue_pending": 0,
        }
        return json.dumps(status, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def register_tools(mcp):
    for func in [vision_describe, vision_compare, image_generate, image_edit, get_status]:
        mcp.tool()(func)
