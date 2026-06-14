from loguru import logger
from PIL import Image

from mcp_server.config import settings


class VisionModel:
    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or settings.vision_model_id
        self._processor = None
        self._model = None
        self._device = None
        self._load()

    def _load(self):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        logger.info(f"Loading vision model: {self.model_id}")
        kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if settings.vision_4bit:
            kwargs["quantization_config"] = {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.bfloat16,
                "bnb_4bit_use_double_quant": True,
            }
        if torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"
            kwargs.pop("device_map", None)

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id, **kwargs
        )
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        if not torch.cuda.is_available():
            self._model = self._model.to(self._device)
        self._model.eval()
        logger.info(f"Vision model {self.model_id} loaded on {self._device}")

    def describe(self, image: Image.Image, prompt: str) -> str:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text.strip()

    def compare(self, image_a: Image.Image, image_b: Image.Image, prompt: str) -> str:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_a},
                    {"type": "image", "image": image_b},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=[image_a, image_b], padding=True, return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text.strip()
