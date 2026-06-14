import subprocess
import sys
import threading
import time

import requests
from loguru import logger
from mcp.server.fastmcp import FastMCP

from mcp_server.tools import register_tools

logger.remove()
fmt = "{time:HH:mm:ss} | {level: <8} | {name} | {message}"
logger.add(sys.stderr, format=fmt, level="INFO")


LITERT_PORT = 9380
LITERT_BIN = "/mnt/juegos/Proyectos/local-multimodal-mcp/.venv/bin/litert-lm"
LITERT_PROC: subprocess.Popen | None = None


def _start_litert_server():
    global LITERT_PROC
    if LITERT_PROC is not None:
        return
    logger.info("Pre-starting LiteRT server...")
    LITERT_PROC = subprocess.Popen(
        [LITERT_BIN, "serve", "--port", str(LITERT_PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    def _wait_ready():
        base_url = f"http://localhost:{LITERT_PORT}"
        for attempt in range(30):
            try:
                resp = requests.get(f"{base_url}/v1/models", timeout=2)
                if resp.status_code == 200:
                    logger.info("LiteRT server ready")
                    return
            except requests.ConnectionError:
                pass
            time.sleep(1)
        logger.warning("LiteRT server may not be ready")

    threading.Thread(target=_wait_ready, daemon=True).start()


def _wait_litert_ready():
    base_url = f"http://localhost:{LITERT_PORT}"
    for attempt in range(30):
        try:
            resp = requests.get(f"{base_url}/v1/models", timeout=2)
            if resp.status_code == 200:
                logger.info("LiteRT server confirmed ready")
                return
        except requests.ConnectionError:
            pass
        time.sleep(1)
    logger.warning("LiteRT server may not be ready")


def _stop_litert_server():
    global LITERT_PROC
    if LITERT_PROC is not None:
        LITERT_PROC.kill()
        LITERT_PROC.wait()
        LITERT_PROC = None
        logger.info("LiteRT server stopped")


# Start LiteRT eagerly at import time
_start_litert_server()


mcp = FastMCP(
    "local-multimodal-mcp",
    instructions="MCP server for local vision and image generation",
)


def main():
    _start_litert_server()
    # Wait for LiteRT server to be ready before accepting client requests
    _wait_litert_ready()
    register_tools(mcp)
    logger.info("local-multimodal-mcp starting (lazy load: models loaded on first use)")
    try:
        mcp.run(transport="stdio")
    finally:
        _stop_litert_server()


if __name__ == "__main__":
    main()
