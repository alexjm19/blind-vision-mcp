# Contributing

Thanks for your interest in contributing!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/alexjm19/local-multimodal-mcp.git`
3. Install dependencies: `uv sync`
4. Create a branch: `git checkout -b feature/my-feature`

## Development

### Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting:

```bash
uv run ruff check .
```

### Testing

Run tests with pytest:

```bash
uv run pytest tests/ -v
```

## Pull Request Process

1. Run `uv run ruff check .` — no errors
2. Run `uv run pytest tests/ -v` — all passing
3. Open a PR against the `main` branch
4. Describe what your PR does and why

## License Compliance

All models and dependencies must have **permissive licenses** (Apache 2.0, MIT, BSD).
No copyleft, GPL, or non-commercial licenses are allowed.

## Questions?

Open an issue on GitHub.
