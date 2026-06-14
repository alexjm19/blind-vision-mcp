# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-06-14

### Added
- Initial release
- `vision_describe` tool with Qwen2-VL-7B-Instruct (Apache 2.0)
- `vision_compare` tool for structured image comparison
- `image_generate` tool with FLUX.1-schnell (Apache 2.0)
- `image_edit` tool with FLUX.1 Kontext Dev (Apache 2.0)
- `get_status` tool for VRAM monitoring
- Automatic VRAM swap via ModelManager (LRU-1 slot)
- Fallback to Qwen2-VL-2B on OOM
- Prompt optimization engine for quality generation
- Feather masking for smooth inpainting edges
- Image loading from local paths and URLs
- Lazy loading — models load on first use, server starts in < 1s
- MIT license — fully open source
- uv-based project management (Python 3.11+)
- GitHub Actions CI (ruff lint + pytest)
