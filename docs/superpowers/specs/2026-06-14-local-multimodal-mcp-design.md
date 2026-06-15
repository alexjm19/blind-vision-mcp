# blind-vision-mcp Design

## Purpose
MCP server that gives vision + image generation to text-only LLMs.
100% local. No API keys. No cloud.

Primary client: **OpenCode** — grant vision to DeepSeek v4 and other
text-only models.

## Architecture
```
Cliente MCP (OpenCode/Claude/Cursor/Cline)
        |  stdin/stdout JSON-RPC
        v
+-------------------------------+
| mcp_server/server.py          |  Entry point (stdio)
| mcp_server/tools.py           |  5 tool handlers
| mcp_server/config.py          |  Paths, model IDs, thresholds
+-------------------------------+
| models/manager.py             |  ModelManager (swap LRU-1)
| models/vision.py              |  Qwen2-VL-7B (fallback 2B)
| models/generation.py          |  FLUX.1-schnell + Kontext Dev
+-------------------------------+
| utils/prompt_optimizer.py     |  Prompt quality enhancement
| utils/image_utils.py          |  Load/resize/feather/mask
+-------------------------------+
```

## VRAM Strategy
- Qwen2-VL-7B (4-bit) ~6 GB
- FLUX.1-schnell (FP8) ~8 GB
- FLUX.1 Kontext Dev ~8 GB
- Cannot fit together in 12 GB
- **Solution**: single-slot LRU, lazy load, auto-swap

## Models & Licenses
| Model | License | Purpose |
|-------|---------|---------|
| Qwen2-VL-7B-Instruct | Apache 2.0 | Image understanding |
| Qwen2-VL-2B-Instruct | Apache 2.0 | Fallback |
| FLUX.1-schnell | Apache 2.0 | Image generation |
| FLUX.1 Kontext Dev | Apache 2.0 | Image editing |

## Tools
1. **vision_describe**(image_path, detail_level?, focus?) — Qwen2-VL
2. **vision_compare**(image_path_a, image_path_b, focus?) — Qwen2-VL
3. **image_generate**(description, output_path?, params?) — FLUX.1-schnell
4. **image_edit**(image_path, mask_path, description, output_path?, params?) — FLUX.1 Kontext Dev
5. **get_status**() — VRAM + loaded models

## Compliance
- All models Apache 2.0
- All deps MIT/BSD/Apache 2.0
- Project license: MIT

## Project Structure
~25 files. uv-only. Python 3.11+.

## Testing
- pytest + pytest-asyncio
- Models mocked (no GPU needed)
- Ruff linting in CI
