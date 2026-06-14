QUALITY_TOKENS = [
    "masterpiece",
    "best quality",
    "highly detailed",
    "sharp focus",
    "cinematic lighting",
    "vivid colors",
]

STYLE_PREFIX = "A stunning, high-quality photograph of "


def build_generation_prompt(description: str) -> str:
    cleaned = description.strip().rstrip(".,")
    tokens = ", ".join(QUALITY_TOKENS)

    prompt = f"{STYLE_PREFIX}{cleaned}, {tokens}"

    if len(prompt) > 1024:
        prompt = prompt[:1021] + "..."

    return prompt


def build_edit_prompt(description: str) -> str:
    return f"Transform the masked region to: {description.strip()}"
