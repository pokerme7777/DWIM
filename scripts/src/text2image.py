from diffusers import DiffusionPipeline  # type: ignore
import torch
from PIL import Image
from typing import Any, cast


class StableDiffusionXLInterpreter:
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        # We cast because MyPy / PyRight doesn't know that
        # the pipe is callable and has a `to` method.
        self.pipe = cast(Any, pipe)
        self.pipe.to(self.device)

    def __call__(self, prompt: str) -> Image.Image:
        return self.pipe(prompt=prompt).images[0]
