from omegaconf import DictConfig
from loguru import logger
import sys
import hydra
from .pale_giant_utils import MeasureTime
from PIL import Image
from typing import Any, Optional
from pydantic import BaseModel


FLUSH_STRING = """
def execute_command(image) -> str:
    return "<FLUSH>"
"""


class PlannerControllerForDemo:
    def __init__(
        self,
        config: DictConfig,
    ):
        self.config = config
        self.program_generator = None

    def setup(self):
        # This function does some funky stuff like inserting things
        # into the global namespace, so we _don't_ instantiate it
        # automatically.

        # Add a logger that only logs at the INFO level.
        logger.remove()
        logger.add(sys.stdout, level="INFO", colorize=True)

        from src.imagepatch_gqa import ImagePatch, bool_to_yesno  # noqa: F401
        import numpy as np  # noqa: F401

        # Do we... actually need to do this? Or is just importing
        # them enough?
        globals()["ImagePatch"] = ImagePatch
        globals()["bool_to_yesno"] = bool_to_yesno
        globals()["np"] = np
        globals()["BaseModel"] = BaseModel
        globals()["Optional"] = Optional

        with MeasureTime() as elapsed_time:
            hydra.utils.instantiate(self.config.module_provider)
        logger.info("init_module_provider took {} seconds", elapsed_time.elapsed_time)

        with MeasureTime() as elapsed_time:
            logger.info("Instantiating program generator")
            self.program_generator = hydra.utils.instantiate(
                self.config.program_generator
            )
        logger.info(
            "program_generator instantiation took {} seconds", elapsed_time.elapsed_time
        )

    def generate_plan(self, context: str) -> str:
        program = self.program_generator.generate(context)  # type: ignore
        return program

    def execute_plan(self, program: str, image_path: str) -> Any:
        # Clear any previous definitions of execute_command.
        exec(compile(FLUSH_STRING, "<string>", "exec"), globals())
        # Compile the program and execute it, which will define
        # a new function called execute_command.
        code_obj = compile(program, "<string>", "exec")
        exec(code_obj, globals())
        # Load the image and execute the command.
        image = Image.open(image_path).convert("RGB")
        result = execute_command(image)  # type: ignore # noqa: F821
        return result

    def execute_multi_image_plan(self, program: str, image_paths: list[str]) -> Any:
        # Clear any previous definitions of execute_command.
        exec(compile(FLUSH_STRING, "<string>", "exec"), globals())
        # Compile the program and execute it, which will define
        # a new function called execute_command.
        code_obj = compile(program, "<string>", "exec")
        exec(code_obj, globals())
        # Load the images and execute the command.
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        result = execute_command(images)  # type: ignore # noqa: F821
        return result
