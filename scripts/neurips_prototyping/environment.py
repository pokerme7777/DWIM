import ast
import asyncio
import enum
import json
import re
from dataclasses import asdict
from string import Template
from typing import Any, Literal, Optional, Union

import ray
from loguru import logger
from pydantic import BaseModel, Discriminator, Tag, ValidationError
from tornado.ioloop import IOLoop
from typing_extensions import Annotated

from src.pale_giant_utils import (
    extract_python_code_in_triple_backticks,
    parse_outer_tag_of_pseudo_xml,
)

from .jupyter import JupyterKernel, JupyterKernelGatewayWrapper
from .prototype_module_providing import (
    ModuleSpec,
    init_module_provider,
    teardown_module_provider,
)


# Note: The `observation_type` field is used to discriminate between the different
# types of observations. This is necessary because we are using a union type to
# represent the different types of observations that can be returned by the
# environment. This is called a tagged union, and ensures that we can
# deserialize an observation correctly by using the `observation_type` field.
class CodeObservation(BaseModel):
    observation_type: Literal["code_observation"] = "code_observation"
    execution_result: str = ""
    program_state: str = ""


class NonCodeObservation(BaseModel):
    observation_type: Literal["non_code_observation"] = "non_code_observation"
    content: str = ""


class NullObservation(BaseModel):
    observation_type: Literal["null_observation"] = "null_observation"
    pass


Observation = Union[CodeObservation, NonCodeObservation, NullObservation]


# Older versions of the code did not have a discriminator field, so we need to
# add a discriminator function to discriminate between the different types of
# observations. This function is used to determine the type of observation
# when deserializing the observation from JSON. It does so by checking to see
# what fields the observation has.
def discriminate_untagged_observation(
    observation: Any,
) -> Literal["code_observation", "non_code_observation", "null_observation"]:
    if isinstance(observation, dict):
        if "execution_result" in observation:
            return "code_observation"
        if "content" in observation:
            return "non_code_observation"
        return "null_observation"
    elif isinstance(observation, BaseModel):
        if hasattr(observation, "execution_result"):
            return "code_observation"
        if hasattr(observation, "content"):
            return "non_code_observation"
        return "null_observation"
    else:
        raise ValidationError(f"Cannot discriminate observation: {observation}")


# We create a separate type to indicate "old-style" observations which did not have a
# discriminator field. See the pydantic docs here to understand how this works:
# https://docs.pydantic.dev/latest/concepts/unions/#discriminated-unions-with-callable-discriminator
ObservationV1 = Annotated[
    Union[
        Annotated[CodeObservation, Tag("code_observation")],
        Annotated[NonCodeObservation, Tag("non_code_observation")],
        Annotated[NullObservation, Tag("null_observation")],
    ],
    Discriminator(discriminate_untagged_observation),
]


# class SynchronousJupyterKernelWrapper:
#     def __init__(self, gateway_ip_addr: str, gateway_port: str, conv_id: str):
#         self.kernel = JupyterKernel(
#             gateway_ip_addr=gateway_ip_addr, gateway_port=gateway_port, conv_id=conv_id
#         )

#         async def _initialize():
#             await self.kernel.initialize()

#         IOLoop.current().run_sync(_initialize)

#     def execute(self, code):
#         logger.debug(f"Executing code: {code}")

#         async def _execute():
#             result = await self.kernel.execute(code)
#             return result

#         result = IOLoop.current().run_sync(_execute)
#         logger.debug(f"Kernel execution result: {result}")
#         return result

#     def shutdown(self):
#         async def _shutdown():
#             await self.kernel.shutdown_async()

#         IOLoop.current().run_sync(_shutdown)


class SynchronousJupyterKernelWrapper:
    def __init__(self, gateway_ip_addr: str, gateway_port: str, conv_id: str):
        self.kernel = JupyterKernel(
            gateway_ip_addr=gateway_ip_addr, gateway_port=gateway_port, conv_id=conv_id
        )

        self.run_sync(self.initialize_kernel)

    async def initialize_kernel(self):
        await self.kernel.initialize()

    def execute(self, code):
        logger.debug(f"Executing code: {code}")
        result = self.run_sync(lambda: self.kernel.execute(code))
        logger.debug(f"Kernel execution result: {result}")
        return result

    def shutdown(self):
        self.run_sync(self.kernel.shutdown_async)

    def run_sync(self, func):
        # Check if there's a running asyncio event loop
        if asyncio.get_event_loop().is_running():
            # If running inside an environment with an active loop, use asyncio
            return asyncio.run_coroutine_threadsafe(
                func(), asyncio.get_event_loop()
            ).result()
        else:
            # Otherwise, use Tornado's IOLoop
            return IOLoop.current().run_sync(func)


class JupyterVisualProgrammingEnvironmentObservationMarkup(enum.Enum):
    PROGRAM_STATE = "program_state"
    RESULT = "result"


DEFAULT_ALLOWED_NAMES = ["ImagePatch", "bool_to_yesno", "image"]
DEFAULT_RESTRICTED_NAMES = ["get_ipython"]

###important about tools ultilization  fucai
class JupyterVisualProgrammingEnvironment:
    def __init__(
        self,
        gateway_ip_addr: str,
        gateway_port: str,
        module_specs: list[ModuleSpec],
        namespace: Optional[str] = None,
        # See the docstring of `ExecutedWithLimitedNamespace`
        # for an explanation of what these arguments do. You
        # don't have to touch them unless you want to give the
        # agent access to more functions that are not defined
        # on the `ImagePatch` class.
        allowed_names: list[str] = DEFAULT_ALLOWED_NAMES,
        restricted_names: list[str] = DEFAULT_RESTRICTED_NAMES,
        NF_flag: int = 0
    ):
        self.gateway_ip_addr = gateway_ip_addr
        self.gateway_port = gateway_port
        self.kernel: Optional[SynchronousJupyterKernelWrapper] = None
        self.allowed_names = set(allowed_names)
        self.restricted_names = set(restricted_names)
        if namespace is not None:
            for spec in module_specs:
                spec.namespace = namespace
        self.module_specs = module_specs
        self._episode_is_done = False
        self._check_ray_is_initialized()
        # Kill any named actors that might be alive from previous runs
        # that were not properly shut down.
        teardown_module_provider(self.module_specs)
        # Bring up the Ray actors that will provide the modules used by
        # the ImagePatch class.
        init_module_provider(self.module_specs)

        # new visual feedback
        self.NF_flag = NF_flag == 1

    @classmethod
    def init_using_raw_module_specs(
        cls, *args, **kwargs
    ) -> "JupyterVisualProgrammingEnvironment":
        module_specs = kwargs.pop("module_specs", [])
        module_specs = [ModuleSpec(**spec) for spec in module_specs]
        kwargs["module_specs"] = module_specs
        return cls(*args, **kwargs)

    def is_done(self):
        return self._episode_is_done

    @staticmethod
    def parse_action(action: str) -> tuple[str, Optional[str]]:
        return parse_outer_tag_of_pseudo_xml(action)

    def step(self, action: str) -> Observation:
        try:
            action_type, action_value = self.parse_action(action)
        except ValueError:
            return self.handle_unparseable_action(action)
        else:
            logger.info(
                "Received action_type=<{action_type}> with action_value={action_value}",
                action_type=action_type,
                action_value=action_value,
            )
            if action_type == "code":
                return self.handle_code_action(action_value)

            if action_type == "thought":
                return self.handle_thought_action(action_value)

            if action_type == "done":
                return self.handle_done_action(action_value)

            return NonCodeObservation(
                content=f"The action type {action_type} is not a valid action. Valid actions are: <code>, <thought>, <done>."
            )

    def handle_unparseable_action(self, action: str) -> NonCodeObservation:
        logger.warning(
            "The action is not well-formed XML. action: {action}", action=action
        )
        # The XML is not well-formed, we cannot do anything.
        return NonCodeObservation(
            content="""I couldn't understand the action.
Write only a single valid action. 
Valid actions are: <code>, <thought>, <done>.
To execute code, use the <code> tag, like so:
<code>
```python
# your code here
```
</code>
To think, use the <thought> tag, like so:
<thought>...your thought here...</thought>
To end the conversation, use the <done> tag, like so:
<done></done>"""
        )

    def handle_code_action(
        self, action_value: Optional[str]
    ) -> Union[CodeObservation, NonCodeObservation]:
        if action_value is None:
            logger.warning(
                "The received action was parsed as code, but was empty.",
            )
            return NonCodeObservation(
                content="""The action was parsed as code, but was empty. 
When using a code action, you must provide a string of Python code to execute, like so:
<code>
```python
# your code here
```
</code>"""
            )

        # here is a part for executing the python code fucai
        logger.info("Executing code.")

        # The code _may_ be wrapped in triple backticks, which we need to remove.
        if maybe_code := extract_python_code_in_triple_backticks(action_value):
            code = maybe_code
        else:
            code = action_value

        # Check to see if there is any attempt to interact with pip or conda,
        # and if so, block it.
        if re.search(r"\b(pip|conda)\b", code):
            return NonCodeObservation(
                content="""You are not allowed to use pip or conda in this environment, or make modifications to the environment.
Please only use the provided modules and functions."""
            )

        execution_result, program_state = self._execute_code(code)
        logger.info("Code execution result: {}", execution_result)
        return CodeObservation(
            execution_result=execution_result, program_state=program_state
        )

    def handle_thought_action(self, action_value: Optional[str]) -> NullObservation:
        return NullObservation()

    def handle_done_action(self, action_value: Optional[str]) -> NullObservation:
        self._episode_is_done = True
        return NullObservation()

    def _execute_code(self, code: str) -> tuple[str, str]:
        if self.kernel is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        # payload = """eval(compile('{code}', '<string>', 'exec'))""".format(code=code)
        # execution_result = self.kernel.execute(payload)
        # execution_result = self.kernel.execute(f"exec({repr(code)})")
        execution_result = self.kernel.execute(f"executor({repr(code)})")
        # execution_result = self.kernel.execute(f"execute_safe({repr(code)})")
        # program_state = self._capture_kernel_locals()
        program_state = self._capture_executor_locals()
        return execution_result, program_state

    def reset(self, image_path: Optional[str] = None):
        self._episode_is_done = False
        if self.kernel is not None:
            self.kernel.shutdown()

        self._create_new_jupyter_kernel("test")
        self._make_modules_available_in_kernel()

        if image_path is not None:
            self.inject_image(image_path)

        self._init_safe_executor_in_kernel()

    def render(self):
        raise NotImplementedError

    def close(self):
        if self.kernel is not None:
            self.kernel.shutdown()
            self.kernel = None
        teardown_module_provider(self.module_specs)

    def _create_new_jupyter_kernel(self, conv_id: str):
        self.kernel = SynchronousJupyterKernelWrapper(
            gateway_ip_addr=self.gateway_ip_addr,
            gateway_port=self.gateway_port,
            conv_id=conv_id,
        )

    def _check_ray_is_initialized(self):
        if not ray.is_initialized():
            raise RuntimeError("Ray is not initialized. Call ray.init() first.")

    def _make_modules_available_in_kernel(self):
        # First we connect the kernel to Ray.
        self._check_ray_is_initialized()
        if self.kernel is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        self.kernel.execute("""import ray; ray.init(address="auto")""")

        # To make the modules available to the kernel, we serialize the module specs
        # and inject them into the kernel as Python dictionaries. Then, the function
        # init_module_provider_in_kernel will be called in the kernel to hook up the
        # Ray actors implementing the modules to the ImagePatch class.
        serialized_module_specs = [asdict(spec) for spec in self.module_specs]  # fucai load the imagepatch for each kernel first

        if self.NF_flag:
            payload = """from src.imagepatch_gqa_NF import ImagePatch, bool_to_yesno, select_best_match_patch_by_description
from neurips_prototyping.prototype_module_providing import init_module_provider_in_kernel 
from src.pale_giant_utils import ExecWithLimitedNamespace
executor = ExecWithLimitedNamespace(allowed_names={allowed_names}, restricted_names={restricted_names})
serialized_module_specs = {serialized_module_specs}
init_module_provider_in_kernel(serialized_module_specs)""".format(
            serialized_module_specs=serialized_module_specs,
            allowed_names=repr(self.allowed_names),
            restricted_names=repr(self.restricted_names),
        )
        else:
            payload = """from src.imagepatch_gqa import ImagePatch, bool_to_yesno, select_best_match_patch_by_description
from neurips_prototyping.prototype_module_providing import init_module_provider_in_kernel 
from src.pale_giant_utils import ExecWithLimitedNamespace
executor = ExecWithLimitedNamespace(allowed_names={allowed_names}, restricted_names={restricted_names})
serialized_module_specs = {serialized_module_specs}
init_module_provider_in_kernel(serialized_module_specs)""".format(
            serialized_module_specs=serialized_module_specs,
            allowed_names=repr(self.allowed_names),
            restricted_names=repr(self.restricted_names),
        )
        if self.kernel is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        return self.kernel.execute(payload)

    def _init_safe_executor_in_kernel(self):
        if self.kernel is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        payload = """from src.pale_giant_utils import ExecWithLimitedNamespace
executor = ExecWithLimitedNamespace(
    allowed_names={allowed_names}, 
    restricted_names={restricted_names},
    inherited_scope=locals()
    )""".format(
            allowed_names=repr(self.allowed_names),
            restricted_names=repr(self.restricted_names),
        )
        return self.kernel.execute(payload)

    def inject_image(self, image_path):
        if self.kernel is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        return self.kernel.execute(
            "from PIL import Image; image = Image.open('{}').convert('RGB')".format(
                image_path
            )
        )

    def _capture_kernel_locals(self) -> str:
        if self.kernel is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        # Avoid noise by not capturing private variables, and also prevent recursively
        # capturing the results of previous captures. This can cause the dictionary to
        # blow up, because it will keep infinitely escaping the captured locals.
        locals_to_ignore = """['In', 'Out', 'get_ipython', 'exit', 'quit', 'open']"""
        serialize_locals_template = Template(
            "import json\n"
            "json.dumps({key: repr(value) for key, value in locals().items()"
            " if not key.startswith('_') and not key in $locals_to_ignore})"
        )
        serialize_locals_snippet = serialize_locals_template.substitute(
            locals_to_ignore=locals_to_ignore
        )
        return self.kernel.execute(serialize_locals_snippet)

    def _capture_executor_locals(self) -> str:
        if self.kernel is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        # Avoid noise by not capturing private variables, and also prevent recursively
        # capturing the results of previous captures. This can cause the dictionary to
        # blow up, because it will keep infinitely escaping the captured locals.
        # locals_to_ignore = """['In', 'Out', 'get_ipython', 'exit', 'quit', 'open']"""
        # serialize_locals_template = Template(
        #     "import json\n"
        #     "json.dumps({key: repr(value) for key, value in executor.locals.items()"
        #     " if not key.startswith('_') and not key in $locals_to_ignore})"
        # )
        # serialize_locals_snippet = serialize_locals_template.substitute(
        #     locals_to_ignore=locals_to_ignore
        # )
        return self.kernel.execute("print(executor.serialize())")


def retrieve_final_result_from_trajectory(observations: list[Observation]) -> str:
    code_observations = [_ for _ in observations if isinstance(_, CodeObservation)]
    if not code_observations:
        return "NO_CODE_OBSERVATIONS"

    last_code_observation = code_observations[-1]

    # The program state is sent back from the kernel as an escaped string
    # representation of a dictionary. We can't directly deserialized it because
    # the string is escaped. We first need to unescape it, which we do by using
    # ast.literal_eval — you can think of this almost like printing the string.
    escaped_serialized_program_state: str = last_code_observation.program_state
    try:
        serialized_program_state: str = ast.literal_eval(
            escaped_serialized_program_state
        )
    except SyntaxError:
        return "CANNOT_STRINGIFY_PROGRAM_STATE"

    try:
        deserialized_program_state: dict[str, str] = json.loads(
            serialized_program_state
        )
    except json.JSONDecodeError:
        return "CANNOT_DESERIALIZE_PROGRAM_STATE"
    except TypeError:
        # Maybe the program state is already a dictionary?
        if isinstance(serialized_program_state, dict):
            deserialized_program_state = serialized_program_state
        else:
            return "CANNOT_DESERIALIZE_PROGRAM_STATE"

    maybe_final_answer = deserialized_program_state.get(
        "final_answer", "FINAL_ANSWER_NOT_FOUND"
    )

    return maybe_final_answer


if __name__ == "__main__":
    ray.init(address="auto")
    module_specs = [
        ModuleSpec(
            name="find_in_image",
            namespace="worker_namespace",
            lifetime="detached",
            num_gpus=0.25,
            dotpath_to_constructor="src.object_detection.HfGroundingDino",
            constructor_params={},
        )
    ]
    with JupyterKernelGatewayWrapper(port="8989") as gateway:
        kernel = SynchronousJupyterKernelWrapper(
            gateway_ip_addr=gateway.ip_address,
            gateway_port=gateway.port,
            conv_id="test",
        )
        print(kernel.execute("print('Hello, World #1!')"))
        kernel.shutdown()

        env = JupyterVisualProgrammingEnvironment(
            gateway_ip_addr=gateway.ip_address,
            gateway_port=gateway.port,
            module_specs=module_specs,
        )
        env.reset(image_path="example_data/fruits.png")
        env.step("<code>print('Hello, World #2!')</code>")
        env.step("<thought>I wonder if there's a banana in the image?</thought>")
        env.step("<code>ImagePatch(image).find('banana')</code>")
        env.close()
