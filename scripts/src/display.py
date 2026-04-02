from .instrumentation import FunctionName, MethodName, TracedFunctionCall
from typing import Callable, Any, Union, Literal
from functools import wraps
from PIL import Image

import base64
from io import BytesIO


class BaseFormatterRegistry:
    registry: dict[
        Union[MethodName, FunctionName, Literal["default"]], Callable[..., Any]
    ] = {}

    @classmethod
    def register(cls, name: Union[MethodName, FunctionName]) -> Callable[..., Any]:
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            cls.registry[name] = func
            return func

        return decorator  # type: ignore


def encode_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def render_images_in_html(images) -> str:
    html = '<div style="display: flex; flex-wrap: wrap;">'
    for img in images:
        img_base64 = encode_image_to_base64(img)
        img_html = f'<img src="data:image/png;base64,{img_base64}" style="flex: 1;">'
        html += img_html
    html += "</div>"
    return html


class HTMLFormatter(BaseFormatterRegistry):
    registry: dict[
        Union[MethodName, FunctionName, Literal["default"]], Callable[..., str]
    ] = {}

    @classmethod
    def register(cls, name: Union[MethodName, FunctionName]) -> Callable[..., str]:
        def decorator(func: Callable[..., str]) -> Callable[..., str]:
            cls.registry[name] = func
            return func

        return decorator  # type: ignore

    @classmethod
    def format(cls, traced_call: TracedFunctionCall) -> str:
        formatter = cls.registry.get(traced_call.name)
        if formatter:
            return formatter(traced_call)
        else:
            default_formatter = cls.registry.get(
                "default", lambda _: f"No formatter for {traced_call.name}"
            )
            return default_formatter(traced_call)


@HTMLFormatter.register(MethodName("ImagePatch.find"))
def format_find(traced_call: TracedFunctionCall) -> str:
    _, object_name = traced_call.args
    returned_patches = traced_call.return_value
    images = [_.cropped_image for _ in returned_patches]
    return render_images_in_html(images)
