from typing import Optional, Union, Any, Callable
from PIL import Image, ImageDraw, ImageFont
from .pale_giant_utils import ModuleProvider
from .instrumentation import ModuleTracer
import numpy as np
import magentic
import functools


class ImagePatch:
    def __init__(
        self,
        image: Image.Image,
        left: Optional[int] = None,
        lower: Optional[int] = None,
        right: Optional[int] = None,
        upper: Optional[int] = None,
        category: Optional[str] = None,
    ):
        """Initializes an ImagePatch object by cropping the image at the given
        coordinates and stores the coordinates as attributes. If no coordinates are
        provided, the image is left unmodified, and the coordinates are set to the
        dimensions of the image.
        Parameters
        -------
        image : array_like
            An array-like of the original image.
        left, lower, right, upper : int
            An int describing the position of the (left/lower/right/upper) border of the
             crop's bounding box in the original image.
        category : str
            A string describing the name of the object in the image.
        """
        self.image = image
        # Rectangles are represented as 4-tuples, (x1, y1, x2, y2),
        # with the upper left corner given first. The coordinate
        # system is assumed to have its origin in the upper left corner, so
        # upper must be less than lower and left must be less than right.

        self.left = left if left is not None else 0
        self.lower = lower if lower is not None else image.height
        self.right = right if right is not None else image.width
        self.upper = upper if upper is not None else 0
        self.cropped_image = image.crop((self.left, self.upper, self.right, self.lower))
        self.horizontal_center = (self.left + self.right) / 2
        self.vertical_center = (self.upper + self.lower) / 2
        self.category = category

    @classmethod
    def from_bounding_box(cls, image: Image.Image, bounding_box: dict) -> "ImagePatch":
        """Initializes an ImagePatch object by cropping the image at the given
        coordinates and stores the coordinates as attributes.
        Parameters
        -------
        image : array_like
            An array-like of the original image.
        bounding_box : dict
            A dictionary like {"box": [left, lower, right, upper], "category": str}.
        """
        left, upper, right, lower = bounding_box["box"]
        category = bounding_box.get("category", None)
        return cls(
            image, left=left, upper=upper, lower=lower, right=right, category=category
        )

    @property
    def xyxy_bbox(self) -> tuple[int, int, int, int]:
        return (self.left, self.upper, self.right, self.lower)

    @property
    def area(self) -> int:
        """
        Returns the area of the bounding box.

        Examples
        --------
        >>> # What color is the largest foo?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     foo_patches = image_patch.find("foo")
        >>>     foo_patches.sort(key=lambda x: x.area)
        >>>     largest_foo_patch = foo_patches[-1]
        >>>     return largest_foo_patch.simple_query("What is the color?")
        """
        return (self.right - self.left) * (self.lower - self.upper)

    def find(self, object_name: str) -> list["ImagePatch"]:
        """Returns a list of ImagePatch objects matching object_name contained in the
        crop if any are found.
        Otherwise, returns an empty list.
        Parameters
        ----------
        object_name : str
            the name of the object to be found

        Returns
        -------
        List[ImagePatch]
            a list of ImagePatch objects matching object_name contained in the crop

        Examples
        --------
        >>> # return the foo
        >>> def execute_command(image) -> List[ImagePatch]:
        >>>     image_patch = ImagePatch(image)
        >>>     foo_patches = image_patch.find("foo")
        >>>     return foo_patches
        """
        return find_in_image(self.cropped_image, object_name)

    def exists(self, object_name: str) -> bool:
        """Returns True if the object specified by object_name is found in the image,
        and False otherwise.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.

        Examples
        -------
        >>> # Are there both foos and garply bars in the photo?
        >>> def execute_command(image)->str:
        >>>     image_patch = ImagePatch(image)
        >>>     is_foo = image_patch.exists("foo")
        >>>     is_garply_bar = image_patch.exists("garply bar")
        >>>     return bool_to_yesno(is_foo and is_garply_bar)
        """
        return len(self.find(object_name)) > 0

    def verify_property(self, object_name: str, visual_property: str) -> bool:
        """
        Returns True if the object possesses the visual property, and False otherwise.
        Differs from 'exists' in that it presupposes the existence of the object s
        pecified by object_name, instead checking whether the object possesses
        the property.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        visual_property : str
            String describing the simple visual property (e.g., color, shape, material)
            to be checked.

        Examples
        -------
        >>> # Do the letters have blue color?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     letters_patches = image_patch.find("letters")
        >>>     # Question assumes only one letter patch
        >>>     return bool_to_yesno(letters_patches[0].verify_property("letters", "blue"))
        """
        return verify_property(self.cropped_image, object_name, visual_property)

    def best_description_from_options(self, object_name: str, property_list: list) -> str:
        """
        Returns best description option from the list about the object.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        property_list : list
            A list of string describing the simple visual property (e.g., color, shape, material)
            to be checked.

        Examples
        -------
        >>> # Are these cats at home or in a zoo?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     return image_patch.best_description_from_options("cats", ["at home", "in a zoo"])
        """
        return best_description_from_options(self.cropped_image, object_name, property_list)

    def simple_query(self, question: Optional[str]) -> str:
        """
        Returns the answer to a basic question asked about the image.
        If no question is provided, returns the answer to "What is this?".
        The questions are about basic perception, and are not meant to be used for
        complex reasoning or external knowledge.
        Parameters
        -------
        question : str
            A string describing the question to be asked.

        Examples
        -------

        >>> # Which kind of baz is not fredding?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     baz_patches = image_patch.find("baz")
        >>>     for baz_patch in baz_patches:
        >>>         if not baz_patch.verify_property("baz", "fredding"):
        >>>             return baz_patch.simple_query("What is this baz?")

        >>> # What color is the foo?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     foo_patches = image_patch.find("foo")
        >>>     foo_patch = foo_patches[0]
        >>>     return foo_patch.simple_query("What is the color?")

        >>> # Is the second bar from the left quuxy?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     bar_patches = image_patch.find("bar")
        >>>     bar_patches.sort(key=lambda x: x.horizontal_center)
        >>>     bar_patch = bar_patches[1]
        >>>     return bar_patch.simple_query("Is the bar quuxy?")
        """
        return simple_query(self.cropped_image, question)

    def visualize(self) -> Image.Image:
        """
        Visualizes the bounding box on the original image and annotates it with the category name if provided.
        """
        canvas = self.image.copy()
        draw = ImageDraw.Draw(canvas)
        box = (self.left, self.upper, self.right, self.lower)

        # Draw the bounding box
        draw.rectangle(box, outline="red", width=2)

        # Write the category name if provided
        if self.category:
            font_size = 20
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()  # type: ignore

            text_position = (self.left, self.upper - font_size)
            draw.text(text_position, self.category, fill="red", font=font)

        return canvas

    def crop_left_of_bbox(
        self, left: int, upper: int, right: int, lower: int
    ) -> "ImagePatch":
        """
        Returns an ImagePatch object representing the area to the left of the given
        bounding box coordinates.

        Parameters
        ----------
        left, upper, right, lower : int
            The coordinates of the bounding box.

        Returns
        -------
        ImagePatch
            An ImagePatch object representing the cropped area.

        Examples
        --------
        >>> # Is the bar to the left of the foo quuxy?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     foo_patch = image_patch.find("foo")[0]
        >>>     left_of_foo_patch = foo_patch.crop_left_of_bbox(
        >>>         foo_patch.left, foo_patch.upper, foo_patch.right, foo_patch.lower
        >>>     )
        >>>     return bool_to_yesno(left_of_foo_patch.verify_property("bar", "quuxy"))
        """
        new_left = 0
        new_upper = 0
        new_right = left
        new_lower = self.cropped_image.size[1]
        new_patch = ImagePatch(
            self.cropped_image,
            left=new_left,
            upper=new_upper,
            lower=new_lower,
            right=new_right,
        )
        return new_patch

    def crop_right_of_bbox(
        self, left: int, upper: int, right: int, lower: int
    ) -> "ImagePatch":
        """
        Returns an ImagePatch object representing the area to the right of the given
        bounding box coordinates.

        Parameters
        ----------
        left, upper, right, lower : int
            The coordinates of the bounding box.

        Returns
        -------
        ImagePatch
            An ImagePatch object representing the cropped area.

        Examples
        --------
        >>> # Is the bar to the right of the foo quuxy?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     foo_patch = image_patch.find("foo")[0]
        >>>     right_of_foo_patch = foo_patch.crop_right_of_bbox(
        >>>         foo_patch.left, foo_patch.upper, foo_patch.right, foo_patch.lower
        >>>     )
        >>>     return bool_to_yesno(right_of_foo_patch.verify_property("bar", "quuxy"))
        """
        new_left = right
        new_upper = 0
        new_right = self.cropped_image.size[0]
        new_lower = self.cropped_image.size[1]
        new_patch = ImagePatch(
            self.cropped_image,
            left=new_left,
            upper=new_upper,
            lower=new_lower,
            right=new_right,
        )
        return new_patch

    def crop_below_bbox(
        self, left: int, upper: int, right: int, lower: int
    ) -> "ImagePatch":
        """
        Returns an ImagePatch object representing the area below the given
        bounding box coordinates.

        Parameters
        ----------
        left, upper, right, lower : int
            The coordinates of the bounding box.

        Returns
        -------
        ImagePatch
            An ImagePatch object representing the cropped area.

        Examples
        --------
        >>> # Is the bar below the foo quuxy?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     foo_patch = image_patch.find("foo")[0]
        >>>     below_foo_patch = foo_patch.crop_below_bbox(
        >>>         foo_patch.left, foo_patch.upper, foo_patch.right, foo_patch.lower
        >>>     )
        >>>     return bool_to_yesno(below_foo_patch.verify_property("bar", "quuxy"))
        """
        new_left = 0
        new_upper = lower
        new_right = self.cropped_image.size[0]
        new_lower = self.cropped_image.size[1]
        new_patch = ImagePatch(
            self.cropped_image,
            left=new_left,
            upper=new_upper,
            lower=new_lower,
            right=new_right,
        )
        return new_patch

    def crop_above_bbox(
        self, left: int, upper: int, right: int, lower: int
    ) -> "ImagePatch":
        """
        Returns an ImagePatch object representing the area above the given
        bounding box coordinates.

        Parameters
        ----------
        left, upper, right, lower : int
            The coordinates of the bounding box.

        Returns
        -------
        ImagePatch
            An ImagePatch object representing the cropped area.

        Examples
        --------
        >>> # Is the bar above the foo quuxy?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     foo_patch = image_patch.find("foo")[0]
        >>>     above_foo_patch = foo_patch.crop_above_bbox(
        >>>         foo_patch.left, foo_patch.upper, foo_patch.right, foo_patch.lower
        >>>     )
        >>>     return bool_to_yesno(above_foo_patch.verify_property("bar", "quuxy"))
        """
        new_left = 0
        new_upper = 0
        new_right = self.cropped_image.size[0]
        new_lower = upper
        new_patch = ImagePatch(
            self.cropped_image,
            left=new_left,
            upper=new_upper,
            lower=new_lower,
            right=new_right,
        )
        return new_patch

    def llm_query(self, question: str) -> str:
        """Answers a text question using GPT-3. The input question is always a formatted string with a variable in it.

        Parameters
        ----------
        question: str
            the text question to ask. Must not contain any reference to 'the image' or 'the photo', etc.

        Examples
        --------
        >>> # What is the city this building is in?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     building_patches = image_patch.find("building")
        >>>     building_patch = building_patches[0]
        >>>     building_name = building_patch.simple_query("What is the name of the building?")
        >>>     return building_patch.llm_query(f"What city is {building_name} in?")

        >>> # Who invented this object?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     object_patches = image_patch.find("object")
        >>>     object_patch = object_patches[0]
        >>>     object_name = object_patch.simple_query("What is the name of the object?")
        >>>     return object_patch.llm_query(f"Who invented {object_name}?")
        """
        return llm_query(question)

    def segment(self, object_name: str) -> list[np.ndarray]:
        """Returns a list of boolean segmentation masks for instances of object_name.

        Parameters
        ----------
        object_name : str
            the name of the object to segment

        Returns
        -------
        List[np.ndarray]
            a list of boolean masks for instances of object_name in the image.

        Examples
        --------
        >>> # Segment the zebras
        >>> def execute_command(image) -> list[np.ndarray]:
        >>>     image_patch = ImagePatch(image)
        >>>     zebra_masks = image_patch.segment("zebra")
        >>>     return zebra_masks
        """
        return segment(self.cropped_image, object_name)

    def depth(self) -> np.ndarray:
        """Returns a depth map of the image.

        Returns
        -------
        np.ndarray
            a depth map of the image.

        Examples
        --------
        >>> # What is the depth of the image?
        >>> def execute_command(image) -> float:
        >>>     image_patch = ImagePatch(image)
        >>>     depth_map = image_patch.depth()
        >>>     return depth_map
        """
        return depth(self.cropped_image)

    def complex_query(self, question: str) -> str:
        """Gives a detailed answer to a question about the image.

        Parameters
        ----------
        question : str
            the question to ask

        Returns
        -------
        str
            the answer to the question

        Examples
        --------
        >>> # Provide a detailed description of the image
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     return image_patch.complex_query("Describe the image in detail.")
        """
        return complex_query(self.cropped_image, question)

    def captioning(self, question: str=None) -> str:
        """Gives a detailed captioning about the image.

        Parameters
        ----------
        question : str
            the question to ask

        Returns
        -------
        str
            the answer to the question

        Examples
        --------
        """
        return captioning(self.cropped_image, question)


    def grounded_query(self, question: str, bounding_boxes: list[list[float]]) -> str:
        """
        Answers a question about the image, using bounding boxes to ground the question.

        Parameters
        ----------
        question : str
            the question to ask
        bounding_boxes : list[list[float]]
            a list of bounding boxes, each represented as [left, upper, right, lower]

        Returns
        -------
        str
            the answer to the question

        Examples
        --------
        >>> # Could the fruit have come from the rightmost tree?
        >>> def execute_command(image) -> str:
        >>>     image_patch = ImagePatch(image)
        >>>     fruit_patches = image_patch.find("fruit")
        >>>     tree_patches = image_patch.find("tree")
        >>>     rightmost_tree = max(tree_patches, key=lambda x: x.horizontal_center)
        >>>     return image_patch.grounded_query(
        >>>         "Could the fruit in <boxes> have come from the tree in <boxes>?",
        >>>         [fruit_patches[0].xyxy_bbox, rightmost_tree.xyxy_bbox],
        >>>     )
        """
        # Make the first character of the question lowercase.
        question = question[0].lower() + question[1:]
        # Now append <image> to the beginning of the question.
        # This looks weird, but it is necessary for the grounding to work.
        question = "In <image>, " + question
        return grounded_query(self.cropped_image, question, bounding_boxes)

    def describe(self) -> str:
        try:
            return complex_query(self.cropped_image, "Describe the image.")
        except ValueError:
            return "Unable to describe this patch."

    def instance_segmentation(self) -> list["ImagePatch"]:
        return [ImagePatch(_) for _ in instance_segmentation(self.cropped_image)]

    def places(self, query: str) -> str:
        return places(query)

    def llm_function(self, template: str) -> Any:
        with ModuleTracer() as tracer:
            # The purpose of this is to allow tracing functions we define using
            # `.llm_function` as a decorator. We don't need to use functools.wraps
            # because both tracer.trace and magentic.prompt use functools.wraps.
            def wrapper(func):
                llm_function = transform_func_into_llm_func(template)(func)
                # return tracer.trace(magentic.prompt(template)(func))
                return tracer.trace(llm_function)

        return wrapper

    def text_to_image(self, prompt: str) -> Image.Image:
        return text_to_image(prompt)

    def web_search(self, query: str) -> list[dict[str, str]]:
        return web_search(query)

    def select_best_match_patch_by_description(patches, object_name, property):
        return select_best_match_patch_by_description(patches, object_name, property)

def bool_to_yesno(bool_answer: bool) -> str:
    return "yes" if bool_answer else "no"


def coerce_to_numeric(x: str) -> float:
    """
    This function takes a string as input and returns a float after removing any
    non-numeric characters. If the input string contains a range (e.g. "10-15"), it
    returns the first value in the range.
    """
    return coerce_to_numeric_impl(x)


def find_in_image(image: Image.Image, object_name: str) -> list[ImagePatch]:
    module = ModuleProvider().get_module("find_in_image")
    assert module is not None, "Module not found"

    box_records = module(image, object_name)
    return [ImagePatch.from_bounding_box(image, box) for box in box_records]


def verify_property(image_patch: Image.Image, object_name: str, property: str) -> bool:
    try:
        module = ModuleProvider().get_module("verify_property")
        assert module is not None, "Module not found"
        return module(image_patch, object_name, property)
    except AssertionError:
        module = ModuleProvider().get_module("simple_query")
        assert module is not None, "Module not found"
        assert hasattr(
            module, "verify_property"
        ), "Module does not have verify_property"
        return module.verify_property(
            image=image_patch, object_name=object_name, property_=property
        )

def best_description_from_options(image: Image.Image, object_name: str, property_list: list) -> str:
    module = ModuleProvider().get_module("simple_query")
    assert module is not None, "Module not found"
    return module.best_description_from_options(image=image, object_name=object_name, property_list=property_list)

def select_best_match_patch_by_description(patches, object_name: str, property: str) -> str:
    module = ModuleProvider().get_module("simple_query")
    assert module is not None, "Module not found"
    loss_patch = []
    for patch in patches:
        loss_patch.append(module.img_description_loss(image=patch.cropped_image, object_name=object_name, property=property))
    output_patch_idx = loss_patch.index(min(loss_patch))
    output_patch = patches[output_patch_idx]
    # print(f"Patch (index:{output_patch_idx} in the patches list) is the best patch to match description: '{object_name} {property}'")
    return output_patch

def best_text_match(image_patch: Image.Image, option_list: list[str]) -> str:
    raise NotImplementedError


def simple_query(image_patch: Image.Image, question: Optional[str]) -> str:
    module = ModuleProvider().get_module("simple_query")
    assert module is not None, "Module not found"

    return module(image_patch, question)

def complex_query(image: Image.Image, question: str) -> str:
    module = ModuleProvider().get_module("simple_query")
    assert module is not None, "Module not found"
    return module.complex_query(image=image, question=question)

def captioning(image: Image.Image, question: str) -> str:
    module = ModuleProvider().get_module("simple_query")
    assert module is not None, "Module not found"
    caption_output = module.complex_query(image=image, question=question)
    print(f"{caption_output}.")
    return caption_output


# def complex_query(image: Image.Image, question: str) -> str:
#     try:
#         module = ModuleProvider().get_module("complex_query")
#         assert module is not None, "Module not found"
#         return module(image, question)
#     except AssertionError:
#         module = ModuleProvider().get_module("simple_query")
#         assert module is not None, "Module not found"
#         assert hasattr(
#             module, "verify_property"
#         ), "Module does not have complex_query"
#         return module.predict(img= image, question=question, short_answer=False)

def llm_query(question: str) -> str:
    module = ModuleProvider().get_module("llm_query")
    assert module is not None, "Module not found"
    return module(question)


def process_guesses(question: str, guesses: list[str]) -> str:
    module = ModuleProvider().get_module("process_guess")
    assert module is not None, "Module not found"
    return module(question, guesses)


def segment(image: Image.Image, object_name: str) -> list[np.ndarray]:
    module = ModuleProvider().get_module("segmenter")
    assert module is not None, "Module not found"
    return module(image, object_name)


def depth(image: Image.Image) -> np.ndarray:
    module = ModuleProvider().get_module("depth_estimator")
    assert module is not None, "Module not found"
    return module(image)


def grounded_query(
    image: Image.Image, question: str, bounding_boxes: list[list[float]]
) -> str:
    module = ModuleProvider().get_module("grounded_query")
    assert module is not None, "Module not found"
    return module(image=image, user_input=question, bounding_boxes=bounding_boxes)


def instance_segmentation(image: Image.Image) -> list[Image.Image]:
    module = ModuleProvider().get_module("instance_segmentation")
    assert module is not None, "Module not found"
    return module(image)


def places(query: str) -> str:
    module = ModuleProvider().get_module("places")
    assert module is not None, "Module not found"
    return module.run(query)


def distance_impl(patch_a: ImagePatch, patch_b: ImagePatch) -> float:
    raise NotImplementedError


def coerce_to_numeric_impl(x: str) -> float:
    raise NotImplementedError


def best_image_match_impl(
    list_patches: list[ImagePatch], content: list[str], return_index: bool = False
) -> Union[ImagePatch, int]:
    raise NotImplementedError


def transform_func_into_llm_func(template: str) -> Callable[..., Any]:
    llm_function_factory = ModuleProvider().get_module("llm_function_factory")
    assert llm_function_factory is not None, "Module not found"
    return llm_function_factory(template)


def text_to_image(prompt: str) -> Image.Image:
    module = ModuleProvider().get_module("text2image")
    assert module is not None, "Module not found"
    return module(prompt)


def web_search(query: str) -> list[dict[str, str]]:
    module = ModuleProvider().get_module("web_search")
    assert module is not None, "Module not found"
    return module.results(query, 1)
