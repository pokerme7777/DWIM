import logging
from typing import Callable, cast

from loguru import logger
from openai import OpenAI
from openai.types.chat import ChatCompletion
from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from .environment import Observation

stdlib_logger = logging.getLogger(__name__)


class GPTReactAgent:
    def __init__(
        self,
        prompter: Callable[[str], str],
        observation_renderer: Callable[[Observation], str],
        model: str = "gpt-3.5-turbo",
        client_factory: type[OpenAI] = OpenAI,
    ):
        self.state: list[dict[str, str]] = []
        self.model = model
        self.observation_renderer = observation_renderer
        self.client = client_factory(timeout=10)
        self.prompter = prompter

    @retry(
        wait=wait_random_exponential(min=1, max=30),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(stdlib_logger, logging.INFO),
    )
    def get_chat_completion(self) -> ChatCompletion:
        api_response = self.client.chat.completions.create(
            model=self.model, messages=self.state  # type: ignore
        )
        return api_response

    def act(self) -> str:
        api_response = self.get_chat_completion()
        action = api_response.choices[0].message.content
        assert action is not None, "API call did not return an action"
        action = cast(str, action)
        logger.info(f"Agent action: {action}")
        return action

    def update(self, experience: tuple[str, Observation]):
        action, observation = experience
        observation_as_str = self.observation_renderer(observation)
        self.state.append({"role": "assistant", "content": action})
        self.state.append({"role": "user", "content": observation_as_str})

    # def reset(self, query: str):
    #     prompt = self.prompter(query)
    #     self.state = [{"role": "system", "content": prompt}]
    def reset(self, query: str, caption_observation:str="", refered_answer:str=""):
        prompt = self.prompter(query, caption_observation, refered_answer)
        self.state = [{"role": "user", "content": prompt}]


if __name__ == "__main__":
    from neurips_prototyping.environment import (
        CodeObservation,
        JupyterVisualProgrammingEnvironment,
        NullObservation,
    )
    from neurips_prototyping.observation_renderers import SimpleXmlRenderer
    from src.prompters import InsertQueryHerePrompter

    prompter = InsertQueryHerePrompter("prompts/2024-04-15-prototype.xml")
    renderer = SimpleXmlRenderer()
    agent = GPTReactAgent(prompter=prompter, observation_renderer=renderer, model='gpt-4o-2024-08-06') #gpt-4o-mini-2024-07-18
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
                execution_result="Code executed successfully with no output"
            ),
        )
    )
    action = agent.act()
    action_type, action_value = JupyterVisualProgrammingEnvironment.parse_action(action)
    print(action_type)
    print(action_value)
