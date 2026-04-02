from transformers import DPTImageProcessor, DPTForDepthEstimation
from PIL import Image
import torch
import numpy as np
from typing import Optional


class DepthEstimator:
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
        self.model.to(self.device)  # type: ignore
        self.model.eval()  # type: ignore

    def predict_depth(self, image: Image.Image) -> tuple[np.ndarray, Image.Image]:
        # MyPy does not correctly infer the type of self.model and self.processor.
        # So we put this guard so that we can use the type checker.
        # typing.cast does not work here.
        assert isinstance(self.model, DPTForDepthEstimation)
        assert isinstance(self.processor, DPTImageProcessor)

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            raw_depth_map = outputs.predicted_depth

        # interpolate to original size
        resized_depth_map = torch.nn.functional.interpolate(
            raw_depth_map.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        # visualize the prediction
        depth_map = resized_depth_map.squeeze().cpu().numpy()
        depth_map_bw = (depth_map * 255 / np.max(depth_map)).astype("uint8")
        depth_map_bw = Image.fromarray(depth_map_bw)
        return depth_map, depth_map_bw

    def __call__(self, image: Image.Image) -> np.ndarray:
        depth_map, _ = self.predict_depth(image)
        return depth_map
