import asyncio
import gc
import threading
import time
from collections.abc import Awaitable
from typing import Callable, Optional

import torch
from loguru import logger

from mcp_server.config import settings


ProgressCb = Callable[[str], Awaitable[None]]
IDLE_TIMEOUT = 60


class ModelManager:
    _loaded_model: Optional[str] = None
    _last_used: float = 0.0
    _vision_model = None
    _generation_model = None
    _edit_model = None

    def __init__(self):
        self._start_idle_monitor()

    def _start_idle_monitor(self):
        def _check():
            while True:
                time.sleep(30)
                if self._loaded_model and time.time() - self._last_used > IDLE_TIMEOUT:
                    logger.info(f"Model idle for {IDLE_TIMEOUT}s, unloading...")
                    try:
                        loop = asyncio.new_event_loop()
                        loop.run_until_complete(self._unload_current())
                        loop.close()
                    except Exception as e:
                        logger.warning(f"Idle unload failed: {e}")

        threading.Thread(target=_check, daemon=True).start()

    async def get_vision_model(self, progress_cb: Optional[ProgressCb] = None):
        self._last_used = time.time()
        if self._loaded_model == "vision" and self._vision_model is not None:
            logger.debug("Vision model already loaded, reusing")
            return self._vision_model

        await self._unload_current(progress_cb)

        msg = "Loading vision model Gemma 4 E2B via LiteRT... ~2.6 GB"
        logger.info(msg)
        if progress_cb:
            await progress_cb(msg)

        try:
            from mcp_server.server import _start_litert_server

            _start_litert_server()
            from models.litert_vision import LiteRTVisionModel

            self._vision_model = LiteRTVisionModel()
            self._loaded_model = "vision"
            msg = "Vision model loaded successfully"
            logger.info(msg)
            if progress_cb:
                await progress_cb(msg)
            return self._vision_model
        except Exception as e:
            logger.error(f"Failed to load vision model: {e}")
            raise RuntimeError(f"Failed to load vision model: {e}") from e

    async def get_generation_model(self, progress_cb: Optional[ProgressCb] = None):
        self._last_used = time.time()
        if self._loaded_model == "generation" and self._generation_model is not None:
            logger.debug("Generation model already loaded, reusing")
            return self._generation_model

        await self._unload_current(progress_cb)
        msg = "Loading generation model FLUX.1-schnell (FP8)... First run downloads ~8 GB from HuggingFace"
        logger.info(msg)
        if progress_cb:
            await progress_cb(msg)

        from models.generation import GenerationModel

        self._generation_model = GenerationModel()
        self._loaded_model = "generation"
        msg = "Generation model loaded successfully"
        logger.info(msg)
        if progress_cb:
            await progress_cb(msg)
        return self._generation_model

    async def get_edit_model(self, progress_cb: Optional[ProgressCb] = None):
        self._last_used = time.time()
        if self._loaded_model == "edit" and self._edit_model is not None:
            logger.debug("Edit model already loaded, reusing")
            return self._edit_model

        await self._unload_current(progress_cb)
        msg = "Loading edit model FLUX.1 Kontext Dev (FP8)... First run downloads ~8 GB from HuggingFace"
        logger.info(msg)
        if progress_cb:
            await progress_cb(msg)

        from models.generation import EditModel

        self._edit_model = EditModel()
        self._loaded_model = "edit"
        msg = "Edit model loaded successfully"
        logger.info(msg)
        if progress_cb:
            await progress_cb(msg)
        return self._edit_model

    async def _unload_current(self, progress_cb: Optional[ProgressCb] = None):
        if self._loaded_model is None:
            return

        msg = f"Unloading {self._loaded_model} model from VRAM to make room..."
        logger.info(msg)
        if progress_cb:
            await progress_cb(msg)

        if self._loaded_model == "vision" and self._vision_model is not None:
            self._vision_model.unload()
            self._vision_model = None
            from mcp_server.server import _stop_litert_server

            _stop_litert_server()
        elif self._loaded_model == "generation" and self._generation_model is not None:
            self._generation_model = None
        elif self._loaded_model == "edit" and self._edit_model is not None:
            self._edit_model = None

        self._loaded_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        msg = "VRAM cleared, model unloaded"
        logger.info(msg)
        if progress_cb:
            await progress_cb(msg)

    async def unload_all(self):
        await self._unload_current()
        self._vision_model = None
        self._generation_model = None
        self._edit_model = None
        self._loaded_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_vram_stats(self) -> dict:
        if not torch.cuda.is_available():
            return {"cuda_available": False, "message": "No CUDA GPU detected"}

        props = torch.cuda.get_device_properties(0)
        total = props.total_memory
        free = torch.cuda.mem_get_info()[0]
        used = total - free
        return {
            "cuda_available": True,
            "device": props.name,
            "total_gb": round(total / 1e9, 2),
            "used_gb": round(used / 1e9, 2),
            "free_gb": round(free / 1e9, 2),
            "loaded_model": self._loaded_model,
        }


model_manager = ModelManager()
