from dataclasses import asdict, dataclass
from typing import Any, Callable, Union

import hydra
import ray
from icecream import ic  # type: ignore[import-untyped]
from loguru import logger
from PIL import Image
from ray.actor import ActorHandle
from tornado.ioloop import IOLoop

from src.imagepatch_gqa import ImagePatch
from src.pale_giant_utils import ModuleProvider

from .jupyter import JupyterKernel, JupyterKernelGatewayWrapper


class ModuleWrapper:
    def __init__(self, module: Union[Callable, ActorHandle]):
        self.module = module

    def __call__(self, *args, **kwargs) -> Any:
        if isinstance(self.module, ActorHandle):
            # If it's a Ray Actor, use remote invocation
            result = self.module.__call__.remote(*args, **kwargs)
            return ray.get(result)
        else:
            # If it's a regular callable object (e.g., function or class with __call__), call it directly
            return self.module(*args, **kwargs)

    # Most of the modules are invoked through a __call__ method.
    # However, we sometimes have auxillary methods on a module to save memory.
    # For example, the vqa model used for simple_query also supports
    # `verify_property`. Otherwise, we would need to load two copies
    # of the VQA model in memory; one for __call__ and one for verify_property.
    # To accomodate this change, we add a new method to the ModuleWrapper class
    # which will handle the remote invocation of the `verify_property` method.
    def verify_property(self, *args, **kwargs) -> bool:
        if isinstance(self.module, ActorHandle):
            # If it's a Ray Actor, use remote invocation
            result = self.module.verify_property.remote(*args, **kwargs)
            return ray.get(result)
        else:
            # If it's a regular callable object (e.g., function or class with __call
            return self.module.verify_property(*args, **kwargs)  # type: ignore
        
    def best_description_from_options(self, *args, **kwargs) -> bool:
        if isinstance(self.module, ActorHandle):
            # If it's a Ray Actor, use remote invocation
            result = self.module.best_description_from_options.remote(*args, **kwargs)
            return ray.get(result)
        else:
            # If it's a regular callable object (e.g., function or class with __call
            return self.module.best_description_from_options(*args, **kwargs)  # type: ignore

    def img_description_loss(self, *args, **kwargs) -> bool:
        if isinstance(self.module, ActorHandle):
            # If it's a Ray Actor, use remote invocation
            result = self.module.img_description_loss.remote(*args, **kwargs)
            return ray.get(result)
        else:
            # If it's a regular callable object (e.g., function or class with __call
            return self.module.img_description_loss(*args, **kwargs)  # type: ignore

    def complex_query(self, *args, **kwargs) -> bool:
        if isinstance(self.module, ActorHandle):
            # If it's a Ray Actor, use remote invocation
            result = self.module.complex_query.remote(*args, **kwargs)
            return ray.get(result)
        else:
            # If it's a regular callable object (e.g., function or class with __call
            return self.module.complex_query(*args, **kwargs)  # type: ignore


@dataclass
class ModuleSpec:
    name: str
    namespace: str
    lifetime: str
    num_gpus: float
    dotpath_to_constructor: str
    constructor_params: dict


def init_module_provider(module_specs: list[ModuleSpec]):
    provider = ModuleProvider()

    for module_spec in module_specs:
        constructor_cls = hydra.utils.get_class(module_spec.dotpath_to_constructor)
        RemoteConstructor = ray.remote(
            constructor_cls
        ).options(  # type:ignore[attr-defined]
            num_gpus=module_spec.num_gpus,
            name=module_spec.name,
            namespace=module_spec.namespace,
            lifetime=module_spec.lifetime,
            max_restarts=3,
        )
        logger.info(
            f"Creating actor {module_spec.name} in namespace {module_spec.namespace}"
        )
        module = RemoteConstructor.remote(**module_spec.constructor_params)
        logger.info(
            f"Created actor {module_spec.name} in namespace {module_spec.namespace}"
        )
        provider.set_module(module_spec.name, ModuleWrapper(module))


def init_module_provider_local(module_specs: list[ModuleSpec]):
    provider = ModuleProvider()

    for module_spec in module_specs:
        constructor_cls = hydra.utils.get_class(module_spec.dotpath_to_constructor)
        module = constructor_cls(**module_spec.constructor_params)
        provider.set_module(module_spec.name, ModuleWrapper(module))


def teardown_module_provider(module_specs: list[ModuleSpec]):
    logger.info("Tearing down module provider")
    provider = ModuleProvider()
    for module_spec in module_specs:
        try:
            actor_handle = ray.get_actor(
                module_spec.name, namespace=module_spec.namespace
            )
        except ValueError:
            logger.warning(
                f"Actor {module_spec.name} not found in namespace {module_spec.namespace}"
            )
            continue
        else:
            ray.kill(actor_handle)
            logger.info(
                f"Killed actor {module_spec.name} in namespace {module_spec.namespace}"
            )
        provider.remove_module(module_spec.name)


def init_module_provider_in_kernel(serialized_module_specs: list[dict]):
    module_specs = [ModuleSpec(**spec) for spec in serialized_module_specs]
    provider = ModuleProvider()

    for module_spec in module_specs:
        actor_handle = ray.get_actor(module_spec.name, namespace=module_spec.namespace)
        provider.set_module(module_spec.name, ModuleWrapper(actor_handle))


if __name__ == "__main__":
    ray_context = ray.init(address="auto")
    module_specs = [
        ModuleSpec(
            name="simple_query",
            namespace="worker_namespace",
            lifetime="detached",
            num_gpus=0.25,
            dotpath_to_constructor="src.vqa.QwenInterpreter",
            constructor_params={"model_slug": "Qwen/Qwen2-VL-2B-Instruct"},
        )
    ]
    init_module_provider(module_specs)
    serialized_module_specs = [asdict(spec) for spec in module_specs]

    image = Image.open("example_data/fruits.png").convert("RGB")
    patch = ImagePatch(image)

    ic(patch.simple_query("what is this?"))

#     ic(patch.find("banana"))

#     preamble = """import ray; ray.init(address='auto')"""
#     run_remote_find_in_image = """from PIL import Image
# from src.imagepatch_gqa import ImagePatch
# from neurips_prototyping.prototype_module_providing import init_module_provider_in_kernel 
# serialized_module_specs = {serialized_module_specs}
# init_module_provider_in_kernel(serialized_module_specs)
# image = Image.open("example_data/fruits.png")
# patch = ImagePatch(image)
# print(patch.find("banana"))""".format(
#         serialized_module_specs=serialized_module_specs
#     )
#     with JupyterKernelGatewayWrapper() as gateway:

#         async def main():
#             kernel = JupyterKernel(
#                 gateway_ip_addr=gateway.ip_address,
#                 gateway_port=gateway.port,
#                 conv_id="test",
#             )
#             await kernel.initialize()
#             result = await kernel.execute("print('Hello, world!')")
#             print(result)
#             result = await kernel.execute("print('1+1=', 1+1)")
#             print(result)
#             result = await kernel.execute(preamble)
#             print(result)
#             result = await kernel.execute(run_remote_find_in_image)
#             print(result)
#             await kernel.shutdown_async()

#         IOLoop.current().run_sync(main)
#     ray.kill(ray.get_actor("find_in_image", namespace="worker_namespace"))
