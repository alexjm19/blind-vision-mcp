# local-multimodal-mcp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development or executing-plans.

**Goal:** Build a complete MCP server with vision + image generation + editing + comparison,
100% local, MIT license, ready for GitHub.

**Architecture:** MCP stdio server with lazy-loaded model swap VRAM management.
Qwen2-VL-7B for vision, FLUX.1-schnell for generation, FLUX.1 Kontext Dev for editing.

**Tech Stack:** Python 3.11, uv, mcp, diffusers, transformers, torch, Pydantic, loguru

---

### Task 1: Write config files

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.gitignore`
- Create: `LICENSE`

### Task 2: Write mcp_server package

**Files:**
- Create: `mcp_server/__init__.py`
- Create: `mcp_server/config.py`
- Create: `mcp_server/tools.py`
- Create: `mcp_server/server.py`

### Task 3: Write models package

**Files:**
- Create: `models/__init__.py`
- Create: `models/manager.py`
- Create: `models/vision.py`
- Create: `models/generation.py`

### Task 4: Write utils package

**Files:**
- Create: `utils/__init__.py`
- Create: `utils/prompt_optimizer.py`
- Create: `utils/image_utils.py`

### Task 5: Write tests

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/test_vision.py`
- Create: `tests/test_generation.py`
- Create: `tests/test_tools.py`

### Task 6: Write docs

**Files:**
- Create: `docs/installation.md`
- Create: `docs/usage.md`
- Create: `docs/compatible_clients.md`

### Task 7: Write GitHub CI and templates

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.github/ISSUE_TEMPLATE/bug_report.md`
- Create: `.github/ISSUE_TEMPLATE/feature_request.md`
- Create: `.github/PULL_REQUEST_TEMPLATE.md`

### Task 8: Write README and root docs

**Files:**
- Create: `README.md`
- Create: `LICENSES_THIRD_PARTY.md`
- Create: `CONTRIBUTING.md`
- Create: `CHANGELOG.md`
