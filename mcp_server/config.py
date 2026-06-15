from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    vision_model_id: str = "litert-community/gemma-4-E2B-it-litert-lm"
    generation_model_id: str = "stabilityai/sdxl-turbo"
    edit_model_id: str = "black-forest-labs/FLUX.1-Kontext-Dev"

    vision_4bit: bool = True
    generation_fp8: bool = True
    edit_fp8: bool = True

    vram_threshold_gb: float = 12.0
    vram_vision_gb: float = 3.0
    vram_generation_gb: float = 8.0
    vram_edit_gb: float = 8.0

    cache_dir: str = str(Path.home() / ".cache" / "huggingface")
    output_dir: str = "outputs"

    default_output_dir: str = field(default_factory=lambda: str(Path.cwd() / "outputs"))

    def __post_init__(self):
        Path(self.default_output_dir).mkdir(parents=True, exist_ok=True)


settings = ModelConfig()
