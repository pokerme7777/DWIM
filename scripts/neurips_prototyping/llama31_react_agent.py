import re
from typing import Callable, Optional, cast

import ray
from loguru import logger
from ray.util.placement_group import (
    placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoTokenizer  # type: ignore
from vllm import LLM, SamplingParams

from .environment import (
    Observation,
)

DEFAULT_SAMPLING_PARAMS = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=300)


class Llama31VllmMReactAgent:
    def __init__(
        self,
        prompter: Callable[[str], str],
        observation_renderer: Callable[[Observation], str],
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        sampling_params: SamplingParams = DEFAULT_SAMPLING_PARAMS,
        vllm_engine_kwargs: Optional[dict] = None,
    ):
        self.state: list[dict[str, str]] = []
        self.model = model
        self.prompter = prompter
        self.observation_renderer = observation_renderer
        if vllm_engine_kwargs is None:
            vllm_engine_kwargs = dict()

        self.llm = LLM(model=self.model, **vllm_engine_kwargs)
        self.sampling_params = sampling_params
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)

    def _do_local_text_generation(self, input_text: str) -> str:
        vllm_output = self.llm.generate([input_text], self.sampling_params)
        generated_text = vllm_output[0].outputs[0].text
        return generated_text

    @staticmethod
    def extract_first_element_skipping_result(text: str):
        # Pattern to match any XML-like element and its content
        pattern = r"<(\w+)>(.*?)</\1>"
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            # Skip the 'result' element
            if match.group(1) == "result":
                continue
            # Return the first non-'result' element found
            element = f"<{match.group(1)}>{match.group(2)}</{match.group(1)}>"
            return element
        return text

    @staticmethod
    def remove_backslashes_from_escaped_underscores(text: str):
        # The model has a tendency to escape underscores with backslashes
        # in the generated text. We remove these backslashes here, because
        # it breaks code execution and is not necessary in "thoughts" anyway.
        return text.replace(r"\_", "_")

    def act(self) -> str:
        input_text = self.tokenizer.apply_chat_template(self.state, tokenize=False)
        input_text = cast(str, input_text)
        output_text = self._do_local_text_generation(input_text)
        action = self.postprocess_action(output_text)
        logger.info(f"Agent action: {action}")
        return action

    def postprocess_action(self, action: str) -> str:
        # The LLM is not guaranteed to stick to the constraint that it only
        # generate one valid XML-like element in an action. We postprocess
        # the generated text here to return the first non-'result' element.
        # TODO: We hardcode that the <result> tag should be skipped here
        # because the LLM should not generate any <result> tag, since this
        # comes from the environment.
        action = self.extract_first_element_skipping_result(action)
        action = self.remove_backslashes_from_escaped_underscores(action)
        return action

    def update(self, experience: tuple[str, Observation]):
        action, observation = experience
        observation_as_str = self.observation_renderer(observation)
        self.state.append({"role": "assistant", "content": action})
        self.state.append({"role": "user", "content": observation_as_str})

    def reset(self, query: str):
        prompt = self.prompter(query)
        self.state = [{"role": "user", "content": prompt}]


class Llama31VllmMReactAgentWithPinnedEngine:
    def __init__(
        self,
        prompter: Callable[[str], str],
        observation_renderer: Callable[[Observation], str],
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        sampling_params: SamplingParams = DEFAULT_SAMPLING_PARAMS,
        vllm_engine_kwargs: Optional[dict] = None,
    ):
        self.state: list[dict[str, str]] = []
        self.model = model
        self.prompter = prompter
        self.observation_renderer = observation_renderer
        if vllm_engine_kwargs is None:
            vllm_engine_kwargs = dict()

        if not vllm_engine_kwargs.get("worker_use_ray"):
            raise ValueError("This agent is not compatible with worker_use_ray=False.")

        tensor_parallel_size = vllm_engine_kwargs.get("tensor_parallel_size")
        if tensor_parallel_size is None:
            raise ValueError("tensor_parallel_size must be specified and > 1")

        if vllm_engine_kwargs.get("pipeline_parallel_size"):
            raise ValueError("pipeline_parallel_size must be 1 for now.")

        # This should be world_size = tensor_parallel_size * pipeline_parallel_size
        # But I ignore the pipeline parallelism for now.
        placement_group_specs = [{"GPU": 1.0} for _ in range(tensor_parallel_size)]
        # Ensure one CPU is available for the driver process.
        placement_group_specs += [{"CPU": 1.0}]
        logger.info(f"Placement group specs: {placement_group_specs}")
        pg = placement_group(placement_group_specs, strategy="STRICT_PACK")

        # Block until the placement group is ready.
        ray.get(pg.ready(), timeout=10)
        logger.info(f"Placement group ready: {pg}")
        llm_constructor = ray.remote(LLM).options(  # type: ignore[attr-defined]
            num_cpus=1,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group_capture_child_tasks=True,
                placement_group=pg,
            ),
        )
        self.llm = llm_constructor.remote(model=self.model, **vllm_engine_kwargs)
        logger.info(f"LLM remote actor created: {self.llm}")
        self.sampling_params = sampling_params
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)

    def _do_local_text_generation(self, input_text: str) -> str:
        vllm_output = ray.get(
            self.llm.generate.remote([input_text], self.sampling_params)
        )
        generated_text = vllm_output[0].outputs[0].text
        return generated_text

    @staticmethod
    def extract_first_element_skipping_result(text: str):
        # Pattern to match any XML-like element and its content
        pattern = r"<(\w+)>(.*?)</\1>"
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            # Skip the 'result' element
            if match.group(1) == "result":
                continue
            # Return the first non-'result' element found
            element = f"<{match.group(1)}>{match.group(2)}</{match.group(1)}>"
            return element
        return text

    @staticmethod
    def remove_backslashes_from_escaped_underscores(text: str):
        # The model has a tendency to escape underscores with backslashes
        # in the generated text. We remove these backslashes here, because
        # it breaks code execution and is not necessary in "thoughts" anyway.
        return text.replace(r"\_", "_")

    def act(self) -> str:
        input_text = self.tokenizer.apply_chat_template(self.state, tokenize=False)
        input_text = cast(str, input_text)
        # print(input_text)
        output_text = self._do_local_text_generation(input_text)
        action = self.postprocess_action(output_text)
        logger.info(f"Agent action: {action}")
        return action

    def postprocess_action(self, action: str) -> str:
        # The LLM is not guaranteed to stick to the constraint that it only
        # generate one valid XML-like element in an action. We postprocess
        # the generated text here to return the first non-'result' element.
        # TODO: We hardcode that the <result> tag should be skipped here
        # because the LLM should not generate any <result> tag, since this
        # comes from the environment.
        action = self.extract_first_element_skipping_result(action)
        action = self.remove_backslashes_from_escaped_underscores(action)
        return action

    def update(self, experience: tuple[str, Observation], add_step_flag=False, step_no=1):
        action, observation = experience
        observation_as_str = self.observation_renderer(observation)
        self.state.append({"role": "assistant", "content": action})
        # add flag if need to add_step
        if add_step_flag:
            observation_as_str=f'{observation_as_str}\nStep {step_no}:'
        self.state.append({"role": "user", "content": observation_as_str})

    # def reset(self, query: str):
    #     prompt = self.prompter(query)
    #     self.state = [{"role": "user", "content": prompt}]
    def reset(self, query: str, caption_observation:str="", refered_answer:str=""):
        prompt = self.prompter(query, caption_observation, refered_answer)
        self.state = [{"role": "user", "content": prompt}]


class StubLocalAgent:
    def __init__(self):
        self.num_invocations = 0

    def act(self):
        if self.num_invocations < 1:
            return f"""<code>print("I have been invoked {self.num_invocations} times.")</code>"""
        else:
            return "<done></done>"

    def update(self, experience):
        self.num_invocations += 1

    def reset(self, query: str):
        self.num_invocations = 0


if __name__ == "__main__":
    logger.enable(__name__)
    from neurips_prototyping.environment import (
        CodeObservation,
        JupyterVisualProgrammingEnvironment,
        NullObservation,
    )
    from neurips_prototyping.observation_renderers import SimpleXmlRenderer
    from src.prompters import InsertQueryHerePrompter

    prompter = InsertQueryHerePrompter("prompts/2024-04-18-mistral-7b.xml")
    observation_renderer = SimpleXmlRenderer()
    agent = Llama31VllmMReactAgentWithPinnedEngine(prompter, observation_renderer, vllm_engine_kwargs={"worker_use_ray": True, "tensor_parallel_size": 1},)
    
    # agent = Llama31VllmMReactAgent(prompter, observation_renderer, vllm_engine_kwargs={"worker_use_ray": True, "tensor_parallel_size": 1},)
    
    agent.reset("How many dogs are brown?")
    agent.update(
        (
            "<thought>I will first find all the dogs, then count the number of dogs which are brown.</thought>",
            NullObservation(),
        )
    )
    agent.update(
        (
            """<code>dog_patches = ImagePatch(image).find("dog")</code>""",
            CodeObservation(
                execution_result="Code executed successfully with no output."
            ),
        )
    )
    action = agent.act()
    action_type, action_value = JupyterVisualProgrammingEnvironment.parse_action(action)
    print(action_type)
    print(action_value)
