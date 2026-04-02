import warnings
from PIL import Image
import numpy as np
import torch

from torchvision.ops import box_convert
from typing import TypedDict
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import logging

import warnings

# Suppress specific UserWarnings from torch.utils.checkpoint
warnings.filterwarnings("ignore", category=UserWarning, module='torch.utils.checkpoint')

class FilterCustomKernelNotLoadedWarning(logging.Filter):
    def filter(self, record):
        return (
            "Could not load the custom kernel for multi-scale deformable attention"
            not in record.getMessage()
        )


# with warnings.catch_warnings():
#     # We are getting a bunch of deprecation warnings from Pydantic in GroundingDINO code.
#     # We don't plan on fixing the GroundingDINO code, so we will suppress these warnings.
#     warnings.filterwarnings("ignore", message=".*smart_union.*", category=UserWarning)
#     warnings.filterwarnings(
#         "ignore", message=".*allow_population_by_field_name.*", category=UserWarning
#     )
#     warnings.filterwarnings(
#         "ignore",
#         message=".*Valid config keys have changed in V2.*",
#         category=UserWarning,
#     )
#     from .panoptic_sam import (  # noqa: E402
#         load_groundingdino,
#         dino_detection,
#     )


class BoxRecord(TypedDict):
    box: list[int]
    category: str
    area: float


class HfGroundingDino:
    def __init__(self, device="cuda:0", image_percentage_threshold: float = 0.01):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.image_percentage_threshold = image_percentage_threshold
        self.model_id = "IDEA-Research/grounding-dino-base"
        # Filter out a warning about a custom CUDA kernel not being
        # loaded that is repeated several times. Perhaps this kernel
        # being missing affects results, but we can't do anything about
        # it right now.
        gdino_logger = logging.getLogger(
            "transformers.models.grounding_dino.modeling_grounding_dino"
        )
        gdino_logger.addFilter(FilterCustomKernelNotLoadedWarning())
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_id
        ).to(self.device)
        # This _specific_ template has to be used for GroundingDino.
        self.template = "a {query}."

    @staticmethod
    def calculate_bbox_area(bbox):
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        return area

    def detect(
        self,
        text: str,
        image: Image.Image,
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
    ) -> list[BoxRecord]:
        inputs = self.processor(
            images=image, text=self.template.format(query=text), return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )

        if len(results) > 1:
            raise ValueError(
                (
                    f"""This code was written assuming that 
                    GroundingDino will only return a list of 
                    length 1 given a single image and text input. 
                    But it returned {len(results)} results: 
                    {results}"""
                )
            )

        gdino_outputs = results[0]
        scores = gdino_outputs["scores"].cpu()  # noqa: F841
        labels = gdino_outputs["labels"]
        boxes = gdino_outputs["boxes"].cpu()

        return [
            BoxRecord(
                box=box.tolist(),
                category=label,
                area=self.calculate_bbox_area(box).item(),
            )
            for box, label in zip(boxes, labels)
        ]

    def __call__(self, image: Image.Image, object_name: str) -> list[BoxRecord]:
        boxes = self.detect(object_name, image)
        return boxes


# class GroundingDinoOVODInterpreter:
#     def __init__(self, image_percentage_threshold=0.00, device="cuda:0"):
#         self.device = device if torch.cuda.is_available() else "cpu"

#         with warnings.catch_warnings():
#             # Ignore the FutureWarning from transformers about the `device` argument.
#             # This comes from GroundingDINO code (I think) and we don't plan on fixing it.
#             # GroundingDINO _might_ stop working in V5 of Transformers.
#             warnings.filterwarnings(
#                 "ignore",
#                 category=FutureWarning,
#                 message=r".*The `device` argument is deprecated.*",
#             )
#             # These are either coming from the GroundingDINO code or are irrelevant HF warnings.
#             warnings.filterwarnings(
#                 "ignore", category=UserWarning, message=r".*torch\.meshgrid.*"
#             )
#             self.groundingdino_model = load_groundingdino(self.device)
#         # We will ignore any objects that are smaller than this percentage of the image.
#         self.image_percentage_threshold = image_percentage_threshold

#     def detect(
#         self, text: str, image: Image.Image
#     ) -> tuple[torch.Tensor, list[str], Image.Image]:
#         dino_box_threshold = 0.3
#         dino_text_threshold = 0.25
#         image_as_array = np.array(image)

#         boxes, categories, visualization = dino_detection(
#             self.groundingdino_model,
#             image,
#             image_as_array,
#             [text],
#             dict(),
#             dino_box_threshold,
#             dino_text_threshold,
#             self.device,
#             visualize=True,
#         )

#         return boxes, categories, visualization

#     @staticmethod
#     def calculate_bbox_area(bbox):
#         x1, y1, x2, y2 = bbox
#         area = (x2 - x1) * (y2 - y1)
#         return area

#     def postprocess_boxes(
#         self, boxes: torch.Tensor, image: Image.Image, categories: list[str]
#     ) -> list[BoxRecord]:
#         image_as_array = np.array(image)
#         h, w, _ = image_as_array.shape
#         boxes = boxes * torch.Tensor([w, h, w, h])
#         xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

#         # Filter out boxes with areas less than 1% of the image size.
#         # image_area = h * w
#         # areas = [self.calculate_bbox_area(box) for box in xyxy]
#         # xyxy = [box for box, area in zip(xyxy, areas) if area / image_area > 0.01]

#         # Convert the xyxy box tensors into a list of dictionaries. This is the
#         # format expected by other parts of the pipeline.
#         boxes_as_records = []
#         for box, category in zip(xyxy, categories):
#             boxes_as_records.append(
#                 BoxRecord(
#                     box=box.astype(int).tolist(),
#                     category=category,
#                     area=self.calculate_bbox_area(box),
#                 )
#             )

#         # Filter out boxes with areas less than 1% of the image size.
#         image_area = h * w
#         boxes_as_records = [
#             box
#             for box in boxes_as_records
#             if box["area"] / image_area > self.image_percentage_threshold
#         ]
#         return boxes_as_records

#     def search_image(
#         self, image: Image.Image, object_name: str
#     ) -> tuple[list[BoxRecord], Image.Image]:
#         raw_boxes, categories, visualization = self.detect(object_name, image)
#         boxes = self.postprocess_boxes(raw_boxes, image, categories)
#         return boxes, visualization

#     def __call__(self, image: Image.Image, object_name: str) -> list[BoxRecord]:
#         with warnings.catch_warnings():
#             # These are either coming from the GroundingDINO code or are irrelevant HF warnings.
#             warnings.filterwarnings(
#                 "ignore", category=UserWarning, message=r".*torch\.meshgrid.*"
#             )
#             warnings.filterwarnings(
#                 "ignore",
#                 category=UserWarning,
#                 message=r".*inputs have requires_grad=True.*",
#             )
#             # Ignore the FutureWarning from transformers about the `device` argument.
#             # This comes from GroundingDINO code (I think) and we don't plan on fixing it.
#             # GroundingDINO _might_ stop working in V5 of Transformers.
#             warnings.filterwarnings(
#                 "ignore",
#                 category=FutureWarning,
#                 message=r".*The `device` argument is deprecated.*",
#             )

#             boxes, _ = self.search_image(image, object_name)
#         return boxes


# class StubObjectDetector:
#     def __init__(self, device="cuda:0"):
#         self.device = device if torch.cuda.is_available() else "cpu"

#     def __call__(self, image: Image.Image, object_name: str) -> list[BoxRecord]:
#         # Choose a random square in the image.
#         w, h = image.size
#         x = np.random.randint(0, w)
#         y = np.random.randint(0, h)
#         box = [x, y, x + 100, y + 100]
#         # Calculate the area of the box.
#         area = (box[2] - box[0]) * (box[3] - box[1])
#         # Return a box record.
#         return [
#             BoxRecord(box=box, category=object_name, area=area),
#         ]


if __name__ == "__main__":
    # detector = GroundingDinoOVODInterpreter()
    detector = HfGroundingDino()
    image = Image.open("fill_the_path_to_example_image.jpg")
    boxes = detector(image, "bowl")
    print("shirt number: ", len(boxes))
    

