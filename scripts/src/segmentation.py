import torch
from .panoptic_sam import (
    load_groundingdino,
    dino_detection,
    load_sam_predictor,
    sam_masks_from_dino_boxes,
)
from PIL import Image
import numpy as np
from torchvision.ops import box_convert  # type: ignore
from typing import Literal, Union, Any, cast, Callable, Optional
from loguru import logger
from transformers import pipeline
from skimage.measure import label, regionprops
from numpy.typing import NDArray


class SamSegmenter:
    def __init__(self, device="cuda:0", groundingdino_model=None):
        # Allow passing in a preloaded model DINO model to save
        # GPU memory. Our open vocab object detector is based
        # on GroundingDINO, so we can use the same model.
        self.device = device
        if groundingdino_model is not None:
            logger.info("Using provided GroundingDINO")
            self.groundingdino_model = load_groundingdino(self.device)
        self.sam_predictor = load_sam_predictor(self.device)
        # NOTE: Bad things may happen if the groundingdino model and the
        # sam_predictor are not on the same device!

    def search_image(
        self, text: str, image: Image.Image
    ) -> tuple[torch.Tensor, list[int], Image.Image]:
        dino_box_threshold = 0.3
        dino_text_threshold = 0.25
        image_as_array = np.array(image)
        boxes, category_ids, visualization = dino_detection(
            self.groundingdino_model,
            image,
            image_as_array,
            [text],
            dict(),
            dino_box_threshold,
            dino_text_threshold,
            self.device,
            visualize=True,
        )

        return boxes, category_ids, visualization

    @staticmethod
    def calculate_bbox_area(bbox: list[float]) -> float:
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        return area

    def postprocess_boxes(
        self, boxes: torch.Tensor, image: Image.Image
    ) -> list[dict[str, list[float]]]:
        image_as_array = np.array(image)
        h, w, _ = image_as_array.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # Convert the xyxy box tensors into a list of dictionaries. This is the
        # format expected by other parts of the pipeline.
        boxes_as_records = []
        for box in xyxy:
            box_as_list: list[float] = box.astype(int).tolist()
            boxes_as_records.append({"box": box_as_list})

        return boxes_as_records

    def do_segmentation(
        self, image: Image.Image, object_name: str
    ) -> list[dict[str, Any]]:
        # Use the DINO model to get the bounding boxes for the object.
        boxes, _, visualization = self.search_image(object_name, image)
        # Feed the boxes into the SAM model to get instance masks.
        masks = self.pred_seg(image, boxes)
        # Postprocess the boxes to convert them into a format we can use
        # for visualization.
        processed_boxes = self.postprocess_boxes(boxes, image)

        # Fold the masks into the postprocessed boxes.
        for mask, box in zip(masks, processed_boxes):
            box.update(mask)

        return processed_boxes

    def __call__(self, image: Image.Image, object_name: str) -> list[np.ndarray]:
        """
        Segments an image using the provided object name.

        Parameters
        -----------
        image: PIL.Image.Image
            The image to segment.
        object_name: str
            The name of the object to segment.

        Returns
        ------------
        list[np.ndarray]
            A list of boolean masks, each of which is a numpy array of shape (H, W)
            where H and W are the height and width of the image.
        """
        boxes = self.do_segmentation(image, object_name)
        masks = [box["mask"] for box in boxes]
        return masks

    def pred_seg(self, img, boxes: torch.Tensor) -> list[dict[str, Any]]:
        """
        Feeds the image and boxes into SAM to get instance masks.

        Parameters
        -----------
        img: PIL.Image.Image
            The image to segment.
        boxes: Tensor[N, 4] where N is the number of boxes and each box is in
            cxcy format with relative coordinates (i.e., values
            between 0 and 1).

        Returns
        ------------
        list[dict[str, Any]]
            A list of dictionaries, each containing a boolean ask and an instance ID.
            The boolean mask is a numpy array of shape (H, W) where H and W are the
            height and width of the image. The instance ID is a unique integer
            identifying the instance.
        """
        self.sam_predictor.set_image(np.array(img))
        thing_masks = sam_masks_from_dino_boxes(
            self.sam_predictor, np.array(img), boxes, self.device
        )

        mask_objects = []
        for instance_id, mask in enumerate(thing_masks):
            mask_objects.append(
                {"mask": mask.squeeze().cpu().numpy(), "inst_id": instance_id}
            )

        return mask_objects


class InstanceSegmentation:
    def __init__(
        self,
        mask_generator: Optional[Callable] = None,
        device: str = "cuda:0",
        overlap_threshold: float = 0.1,
        size_threshold: float = 0.02,
    ):
        self.device = device
        if mask_generator is None:
            self.mask_generator = pipeline(
                "mask-generation", model="facebook/sam-vit-huge", device=self.device
            )
        else:
            self.mask_generator = mask_generator
        self.overlap_threshold = overlap_threshold
        self.size_threshold = size_threshold

    @staticmethod
    def render_cropped_mask(
        mask: NDArray[np.bool_], raw_image: Image.Image
    ) -> Image.Image:
        # Convert image to numpy array
        image_np = np.array(raw_image)

        # Assuming 'mask' is your numpy array
        labeled = label(mask)
        props = regionprops(labeled)

        # Find bounding box of the convex hull
        if not props:
            return Image.fromarray(np.zeros_like(image_np))

        # Find the bounding box of the largest area
        bbox = max(props, key=lambda prop: prop.area).bbox

        # Apply the mask
        result_np = np.where(mask[..., None], image_np, 0)

        # Crop the image
        cropped_image = result_np[bbox[0] : bbox[2], bbox[1] : bbox[3]]

        # Convert back to PIL Image and show
        cropped_pil_image = Image.fromarray(cropped_image)

        return cropped_pil_image

    @staticmethod
    def is_overlapping(mask1, mask2, threshold):
        intersection = np.sum(np.logical_and(mask1, mask2))
        union = np.sum(np.logical_or(mask1, mask2))
        iou = intersection / union
        return iou > threshold

    @staticmethod
    def calculate_mask_area(mask: NDArray[np.bool_]) -> float:
        return float(np.sum(mask) / mask.size)

    def non_maximum_suppression_masks(self, masks):
        if len(masks) == 0:
            return []

        # Compute areas of masks
        areas = np.array([np.sum(mask) for mask in masks])

        # Sort masks by area in descending order
        sorted_indices = np.argsort(-areas)
        masks_sorted = [masks[i] for i in sorted_indices]

        keep = []
        while masks_sorted:
            # Take the mask with the largest area
            current = masks_sorted.pop(0)
            keep.append(current)

            # Compare the current mask with the rest and filter out overlaps
            masks_sorted = [
                mask
                for mask in masks_sorted
                if not self.is_overlapping(current, mask, self.overlap_threshold)
            ]

        # Map the kept masks back to their original order
        keep_indices = [
            np.where([np.array_equal(k, m) for m in masks])[0][0] for k in keep
        ]
        keep_sorted = [masks[i] for i in keep_indices]

        return keep_sorted

    def get_masks(self, image: Image.Image) -> list[NDArray[np.bool_]]:
        outputs = self.mask_generator(image, points_per_batch=64)
        return outputs["masks"]  # type: ignore

    def postprocess_masks(
        self, masks: list[NDArray[np.bool_]]
    ) -> list[NDArray[np.bool_]]:
        masks = self.non_maximum_suppression_masks(masks)
        masks = [
            mask
            for mask in masks
            if self.calculate_mask_area(mask) > self.size_threshold
        ]
        return masks

    def __call__(self, image: Image.Image) -> list[Image.Image]:
        masks = self.get_masks(image)
        masks = self.postprocess_masks(masks)
        cropped_masks = [self.render_cropped_mask(mask, image) for mask in masks]
        # Drop any rendered masks that don't have the right number of channels.
        cropped_masks = [
            mask
            for mask in cropped_masks
            if len(np.array(mask).shape) == 3 and np.array(mask).shape[2] == 3
        ]
        return cropped_masks
