# local-multimodal-mcp

> Give vision and image superpowers to any text-only LLM — 100% local, zero cloud, zero API keys.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)]()
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20≥8GB%20VRAM-green)]()
[![Models](https://img.shields.io/badge/models-Apache%202.0-green)]()
[![Gemma 4](https://img.shields.io/badge/Gemma_4-E2B-purple)]()
[![LiteRT](https://img.shields.io/badge/runtime-LiteRT-orange)]()

---

## What is this?

An **MCP server** that gives any text-only LLM (DeepSeek, Claude, Gemini, etc.) the ability to **see images** and **generate images** — running entirely on your GPU. No data ever leaves your machine.

| Tool | What it does |
|------|-------------|
| `vision_describe` | Analyzes images in extreme detail |
| `vision_compare` | Compares two images side by side |
| `image_generate` | Creates images from text descriptions |
| `image_edit` | Edits specific zones using masks |
| `get_status` | Shows VRAM usage and loaded models |

---

## Why LiteRT + Gemma 4?

Most MCP vision servers use PyTorch transformers with 7B+ param models that consume **6-10 GB VRAM**. This server uses **Google's LiteRT runtime** with **Gemma 4 E2B** — the same stack powering Google's on-device AI Gallery app.

| Metric | Typical PyTorch setup | **This server** |
|--------|---------------------|-----------------|
| Vision VRAM | 6-10 GB | **~2.6 GB** |
| Model size on disk | 7-10 GB | **2.6 GB** |
| Loading spike | Full BF16 (10+ GB) | **None** (pre-quantized) |
| Engine | bitsandbytes + transformers | **LiteRT** (Google) |

The model loads directly in mixed 2/4/8-bit quantization — no "load BF16 first then quantize" VRAM spike.

---

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA ≥8 GB VRAM | NVIDIA ≥12 GB VRAM |
| **RAM** | 16 GB | 32 GB |
| **Storage** | 15 GB free | 30 GB free |
| **CUDA** | 12.x | 12.x |
| **Python** | 3.11 | 3.12 |

**Vision-only** runs on ≥8 GB GPUs. **Vision + Generation** needs ≥12 GB.

---

## Quick Start

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and install
git clone https://github.com/alexjm19/local-multimodal-mcp.git
cd local-multimodal-mcp
uv sync

# 3. Import the vision model (first time only, downloads 2.6 GB)
litert-lm import \
  --from-huggingface-repo litert-community/gemma-4-E2B-it-litert-lm \
  gemma-4-E2B-it.litertlm \
  gemma4-vision

# 4. Run the server
uv run local-multimodal-mcp
```

---

## Architecture

```
┌──────────────┐     JSON-RPC/stdin/stdout     ┌──────────────────────┐
│  MCP Client   │ ◄──────────────────────────► │  mcp_server/server   │
│  (OpenCode,   │                               │  + tools.py          │
│   Claude...)  │                               └──────┬───────────────┘
└──────────────┘                                      │
                                              ┌───────▼───────────────┐
                                              │    ModelManager        │
                                              │    (auto-swap, LRU-1)  │
                                              └───┬──────────┬────────┘
                                        ┌────────▼──┐  ┌────▼──────────┐
                                        │  LiteRT    │  │  FLUX.1       │
                                        │  + Gemma 4 │  │  schnell /    │
                                        │  E2B 2.6GB │  │  Kontext Dev  │
                                        │  (vision)  │  │  (gen/edit)   │
                                        └────────────┘  └───────────────┘
```

All 3 models (~19 GB total) can't fit in 12 GB simultaneously. **ModelManager** handles automatic swap:

| Call | Action |
|------|--------|
| `vision_describe` | Starts LiteRT server → Gemma 4 E2B (~2.6 GB) |
| `image_generate` | Kills LiteRT → Loads FLUX.1-schnell (~8 GB) |
| `image_edit` | Loads FLUX Kontext Dev (~8 GB) |
| `vision_describe` again | Kills FLUX → Restarts LiteRT |

---

## Client Configuration

### OpenCode
```json
{
  "mcpServers": {
    "local-multimodal-mcp": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/local-multimodal-mcp",
        "local-multimodal-mcp"
      ]
    }
  }
}
```

### Claude Desktop / Cursor / Cline
See [installation guide](docs/installation.md).

---

## Usage Examples

```bash
# Describe an image
vision_describe(image_path="/path/to/photo.jpg")

# Compare two images
vision_compare(image_path_a="/path/to/a.png", image_path_b="/path/to/b.png")

# Generate an image
image_generate(description="a mystical forest with glowing mushrooms")

# Edit a masked area
image_edit(image_path="photo.jpg", mask_path="mask.png", description="replace with cherry blossom tree")
```

---

## Models & Licenses

All models are **Apache 2.0** — fully permissive, no restrictions:

| Model | License | Purpose | Size |
|-------|---------|---------|------|
| Gemma 4 E2B (Google) | Apache 2.0 | Image understanding | 2.6 GB |
| FLUX.1-schnell (Black Forest Labs) | Apache 2.0 | Image generation | ~8 GB |
| FLUX.1 Kontext Dev | Apache 2.0 | Image editing | ~8 GB |

---

## Why this matters

Existing MCP vision solutions either:
- **Send your data to the cloud** (OpenAI Vision, Google Vertex)
- **Consume 10+ GB VRAM** (Qwen2-VL-7B, LLaVA)
- **Require complicated setup** (multiple conda envs, manual downloads)

This server gives you **Gemma 4 quality** at **2.6 GB VRAM** with a **single command** install. Powered by Google's LiteRT — the same technology running Gemini Nano on Android.

---

## Star History

If you find this useful, **⭐ star the repo**! It helps others discover local-first AI tools.

---

## License

MIT — see [LICENSE](LICENSE).
