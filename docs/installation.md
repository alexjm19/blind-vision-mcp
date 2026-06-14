# Installation

## Prerequisites

- **NVIDIA GPU** with ≥8 GB VRAM (12 GB recommended for vision + generation)
- **CUDA 12.x** and compatible drivers
- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** package manager

## Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Clone & Setup

```bash
git clone https://github.com/alexjm19/local-multimodal-mcp.git
cd local-multimodal-mcp

uv sync
```

This creates a virtual environment and installs all dependencies.

## Verify Installation

```bash
uv run python -c "from mcp_server.server import main; print('Import OK')"
```

## Client Configuration

### OpenCode

Add to your `opencode.json`:

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

### Claude Desktop

Add to your `claude_desktop_config.json`:

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

### Cursor

In Cursor settings → MCP Servers → Add:

```
Name: local-multimodal-mcp
Type: command
Command: uv run --directory /path/to/local-multimodal-mcp local-multimodal-mcp
```

## First Run

Models are **lazy-loaded** on first use. The first tool call will download the model
from HuggingFace (several GB). Subsequent calls use the cached version.
