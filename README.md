# blind-vision-mcp

> Give vision to any text-only LLM — 100% local, no API costs, your privacy intact.

[![Version](https://img.shields.io/badge/version-0.1.0--beta-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![Python](https://img.shields.io/badge/python-3.11+-blue)]()
[![GPU](https://img.shields.io/badge/GPU-NVIDIA_≥8GB_VRAM-green)]()
[![Gemma 4](https://img.shields.io/badge/vision-Gemma_4_E2B-purple)]()
[![Runtime](https://img.shields.io/badge/runtime-LiteRT-orange)]()

---

## Why this exists

**I use DeepSeek v4 Flash** — an incredible text model. But it's **blind**. It can't see screenshots, images, or UI layouts.

I was tired of:
- Paying **$20-200/month** for vision-capable APIs (GPT-4 Vision, Claude)
- Sending **sensitive screenshots to the cloud**
- **Context switching** between coding and describing images manually

So I built **blind-vision-mcp**: an MCP server that sits between your text LLM and your desktop, letting it "see" through an **on-device vision model** — Google's **Gemma 4 E2B** running via **LiteRT**.

### My specific use case

I control an **Android emulator** that takes **screenshots** of the device. DeepSeek v4 Flash reads those screenshots via blind-vision-mcp and tells the emulator what to do next. It works like this:

```
Emulator takes screenshot → blind-vision-mcp analyzes it with Gemma 4 → 
DeepSeek reads the description → decides next action → ADB command
```

All of this happens **locally**, **privately**, and **without paying per-token API fees**.

---

## What it does

| Capability | Status | Model |
|-----------|--------|-------|
| 👁️ **Image analysis** | ✅ **Stable** | Gemma 4 E2B via LiteRT (~2.6 GB VRAM) |
| 🔄 **Image comparison** | ✅ **Stable** | Gemma 4 E2B via LiteRT |
| 🎨 **Image generation** | 🧪 **Beta** | FLUX.1-schnell (needs HF token) |
| ✏️ **Image editing** | 🧪 **Beta** | FLUX Kontext Dev (needs HF token) |

---

## Key features

- **No API keys needed** for vision — runs 100% on your GPU
- **~2.6 GB VRAM** for vision (not 10+ GB like other solutions)
- **GPU-first** — falls back to CPU if GPU fails
- **Google LiteRT** — same stack powering Gemini Nano on Android
- **Macro-friendly** — perfect for automating emulators, browsers, UIs
- **Works with any MCP client** — OpenCode, Claude Desktop, Cursor, Cline

---

## Quick Start

```bash
# 1. Prerequisites
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and install
git clone https://github.com/alexjm19/blind-vision-mcp.git
cd blind-vision-mcp
uv sync

# 3. Import the vision model (one-time, downloads ~2.6 GB)
litert-lm import \
  --from-huggingface-repo litert-community/gemma-4-E2B-it-litert-lm \
  gemma-4-E2B-it.litertlm \
  gemma4-vision

# 4. Start the server
uv run blind-vision-mcp
```

> **For image generation/editing**: Create a `.env` file with your HF token:
> ```
> HF_TOKEN=hf_your_token_here
> ```
> Then accept terms at https://huggingface.co/black-forest-labs/FLUX.1-schnell

---

## Configuration for OpenCode

Add to your `opencode.json`:

```json
{
  "mcpServers": {
    "blind-vision-mcp": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/blind-vision-mcp",
        "blind-vision-mcp"
      ]
    }
  }
}
```

---

## Usage Examples

```bash
# Analyze a screenshot (perfect for emulator control)
vision_describe(image="/path/to/screenshot.png")

# Compare before/after
vision_compare(image_a="/path/to/before.png", image_b="/path/to/after.png")

# Generate an image (beta)
image_generate(description="a beautiful landscape")

# Check server status
get_status()
```

---

## How vision works (the cool part)

```
┌─────────────────────────────────────────────────────────┐
│  DeepSeek v4 Flash (text-only)                          │
│  "What's on the screen? → vision_describe(screenshot)"  │
└────────────────────────┬────────────────────────────────┘
                         │ MCP protocol (stdin/stdout)
┌────────────────────────▼────────────────────────────────┐
│  blind-vision-mcp server                                 │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ tools.py    │→│ LiteRT server │→│ Gemma 4 E2B     │  │
│  │ (MCP tools) │  │ (port 9380)  │  │ (2.6 GB VRAM)   │  │
│  └────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

The vision model (**Gemma 4 E2B**) runs entirely on your GPU via **Google's LiteRT runtime**. No data ever leaves your machine. The model is pre-quantized (mixed 2/4/8-bit) and loads directly at ~2.6 GB — no "load BF16 first then quantize" memory spike.

---

## Why not just use a vision LLM?

| Solution | Cost | Privacy | VRAM | Quality |
|----------|------|---------|------|---------|
| GPT-4 Vision | $10-20/mo | ❌ Cloud | N/A | Excellent |
| Claude Vision | $20/mo | ❌ Cloud | N/A | Excellent |
| Qwen2-VL-7B (local) | Free | ✅ Local | ~10 GB VRAM | Good |
| **blind-vision-mcp** | **Free** | **✅ Local** | **~2.6 GB VRAM** | **Great** |

---

## Requirements

| Component | Minimum |
|-----------|---------|
| **GPU** | NVIDIA ≥8 GB VRAM (vision) / ≥12 GB (vision + gen) |
| **RAM** | 16 GB |
| **Storage** | 5 GB free for vision model |
| **CUDA** | 12.x |

---

## Project Status

- **Vision**: ✅ Stable and tested
- **Image generation**: 🧪 In beta (needs HF token, FLUX model)
- **Image editing**: 🧪 In beta
- **Version**: 0.1.0 — API may change

---

## License

MIT — see [LICENSE](LICENSE).

---

## Support

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://buymeacoffee.com/alexjmwarea)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=alexjm19/blind-vision-mcp&type=Date)](https://star-history.com/#alexjm19/blind-vision-mcp&Date)

If this saves you from another API bill, **⭐ star the repo**. It helps others find local-first AI tools.

---

Built with ❤️ by [alexjm19](https://github.com/alexjm19)
