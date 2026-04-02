from PIL import Image


def crop_left_of_bbox(image: Image.Image, bbox: list[int]) -> Image.Image:
    """
    Crops the image to the left of the given bounding box.
    """
    left_crop = (0, 0, bbox[0], image.size[1])
    return image.crop(left_crop)


def crop_right_of_bbox(image: Image.Image, bbox: list[int]) -> Image.Image:
    """
    Crops the image to the right of the given bounding box.
    """
    right_crop = (bbox[2], 0, image.size[0], image.size[1])
    return image.crop(right_crop)


def crop_below_bbox(image: Image.Image, bbox: list[int]) -> Image.Image:
    """
    Crops the image below the given bounding box.
    """
    below_crop = (0, bbox[3], image.size[0], image.size[1])
    return image.crop(below_crop)


def crop_above_bbox(image: Image.Image, bbox: list[int]) -> Image.Image:
    """
    Crops the image above the given bounding box.
    """
    above_crop = (0, 0, image.size[0], bbox[1])
    return image.crop(above_crop)
