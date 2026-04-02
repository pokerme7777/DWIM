from typing import Callable
from .imagepatch_gqa import ImagePatch, bool_to_yesno
from .vqa import BlipVqaInterpreter, VerifyPropertyInterpreter, LlavaInterpreter
from .object_detection import GroundingDinoOVODInterpreter, StubObjectDetector
from loguru import logger
from .pale_giant_utils import ModuleProvider
from .openai import ChatGptQuestionAnswer, ChatGptProcessGuess
from .segmentation import SamSegmenter, InstanceSegmentation
from .depth import DepthEstimator
from .shikra import ShikraAskAboutImageInterpreter
from .text2image import StableDiffusionXLInterpreter
import magentic
from langchain.tools import GooglePlacesTool
# from langchain_community.utilities import BingSearchAPIWrapper


def init_module_provider(device="cuda:0"):
    logger.info("Instantiate module provider on {}", device)
    provider = ModuleProvider()
    vqa_model = BlipVqaInterpreter(device=device)
    property_verifier = VerifyPropertyInterpreter(vqa_model)
    open_vocab_object_detector = GroundingDinoOVODInterpreter(device=device)
    llm_query = ChatGptQuestionAnswer()
    process_guess = ChatGptProcessGuess()

    provider.set_module("simple_query", vqa_model)
    provider.set_module("verify_property", property_verifier)
    provider.set_module("find_in_image", open_vocab_object_detector)
    provider.set_module("llm_query", llm_query)
    provider.set_module("process_guess", process_guess)


def init_module_provider_v2(device="cuda:0"):
    logger.info("Instantiate module provider on {}", device)
    provider = ModuleProvider()
    vqa_model = BlipVqaInterpreter(device=device)
    property_verifier = VerifyPropertyInterpreter(vqa_model)
    open_vocab_object_detector = GroundingDinoOVODInterpreter(device=device)
    llm_query = ChatGptQuestionAnswer()
    sam_segmenter = SamSegmenter(
        device=device,
        groundingdino_model=open_vocab_object_detector.groundingdino_model,
    )
    depth_estimator = DepthEstimator(device=device)
    llava = LlavaInterpreter(device=device)
    shikra = ShikraAskAboutImageInterpreter(device=device)
    instance_segmentation = InstanceSegmentation(device=device)
    google_places_tool = GooglePlacesTool()
    # HACK: I added this just to speed up demo development. This
    # should be removed and made configurable.
    stable_diffusion_xl = StableDiffusionXLInterpreter(device="cuda:2")

    provider.set_module("simple_query", vqa_model)
    provider.set_module("verify_property", property_verifier)
    provider.set_module("find_in_image", open_vocab_object_detector)
    provider.set_module("llm_query", llm_query)
    provider.set_module("segmenter", sam_segmenter)
    provider.set_module("depth_estimator", depth_estimator)
    provider.set_module("complex_query", llava)
    provider.set_module("grounded_query", shikra)
    provider.set_module("instance_segmentation", instance_segmentation)
    provider.set_module("places", google_places_tool)
    provider.set_module("llm_function_factory", magentic.prompt)
    provider.set_module("web_search", BingSearchAPIWrapper())
    provider.set_module("text2image", stable_diffusion_xl)


def init_stub_module_provider(device: str = "cuda:0"):
    provider = ModuleProvider()
    open_vocab_object_detector = StubObjectDetector()
    provider.set_module("find_in_image", open_vocab_object_detector)

    def stub_llm_function_factory(template: str) -> Callable:
        def wrapper(func: Callable) -> Callable[..., str]:
            def inner_wrapper(*args, **kwargs):
                return "You asked me to " + template

            return inner_wrapper

        return wrapper

    provider.set_module(
        "llm_function_factory",
        stub_llm_function_factory,
    )
