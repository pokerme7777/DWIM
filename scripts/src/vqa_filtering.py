import torch
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
)
from PIL import Image
from src.pale_giant_utils import JsonlIoHandler
from torch.utils.data import Dataset
import os
import pandas as pd
from tqdm import tqdm

class LlavaInterpreter:
    def __init__(self, device="cuda:0"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_slug = "llava-hf/llava-1.5-7b-hf"
        self.prompt_template = "<image>\nUSER: {user_message}\nASSISTANT:"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_slug,
            low_cpu_mem_usage=True,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_slug)
        self.model.eval()  # type: ignore

    def predict(self, img: Image.Image, question: str) -> str:

        question = f'{question}\nAnswer the question using a single word or phrase.'
        prompt = self.prompt_template.format(user_message=question)
        inputs = self.processor(text=prompt, images=img, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            generate_ids = self.model.generate(**inputs, max_length=200)  # type: ignore

        # Ignore the input tokens and only return the generated text.
        prompt_length = inputs.input_ids.shape[1]
        token_ids_to_decode = generate_ids[:, prompt_length:]
        generated_tokens = self.processor.batch_decode(
            token_ids_to_decode,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return generated_tokens

    def __call__(self, image: Image.Image, question: str) -> str:
        return self.predict(image, question)


class JsonlDatasetWithImageRoot_with_postcaption(Dataset):
    def __init__(self, image_root, jsonl_path):
        self.image_root = image_root
        self.jsonl_path = jsonl_path
        self.records = JsonlIoHandler(self.jsonl_path).read_all()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        record["image_id"] = os.path.join(self.image_root, record["image_id"])
        return record


    