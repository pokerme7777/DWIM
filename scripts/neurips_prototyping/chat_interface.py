import os
import subprocess
from typing import cast

import gradio as gr  # type: ignore
import ray
from omegaconf import OmegaConf

from src.prompters import InsertQueryHerePrompter

from .environment import (
    CodeObservation,
    JupyterVisualProgrammingEnvironment,
    NonCodeObservation,
    NullObservation,
    Observation,
)
from .local_agent import MistralVllmMReactAgentWithPinnedEngine
from .new_worker import Experience
from .observation_renderers import SimpleXmlRenderer
from .prototype_module_providing import ModuleSpec


def get_external_address():
    return subprocess.check_output("hostname -i", shell=True).decode().strip()


GRADIO_SERVER_PORT = 9099

module_specs = [
    ModuleSpec(
        name="find_in_image",
        namespace="interactive",
        lifetime="detached",
        num_gpus=0.125,
        dotpath_to_constructor="src.object_detection.HfGroundingDino",
        constructor_params={},
    ),
    ModuleSpec(
        name="simple_query",
        namespace="interactive",
        lifetime="detached",
        num_gpus=0.125,
        dotpath_to_constructor="src.vqa.BlipVqaInterpreter",
        constructor_params={},
    ),
    ModuleSpec(
        name="llm_query",
        namespace="interactive",
        lifetime="detached",
        num_gpus=0.125,
        dotpath_to_constructor="src.openai.ChatGptQuestionAnswer",
        constructor_params={},
    ),
]


def render_agent_action(action_type: str, action_value: str) -> str:
    match action_type:
        case "code":
            return f"```python\n{action_value}\n```"
        case "thought":
            return f"{action_value}"
        case "done":
            return "I'm done."
        case _:
            return f"{action_type}: {action_value}"


def format_env_observation_for_display(observation: Observation) -> str:
    match observation:
        case CodeObservation():
            return f"{observation.execution_result}"
        case NonCodeObservation():
            return observation.content
        case NullObservation():
            return ""
        case _:
            return "Unknown observation type"


class CurrentTrajectory:
    def __init__(self) -> None:
        self.trajectory: list[Experience] = []

    def add(self, experience: Experience):
        self.trajectory.append(experience)

    def clear(self):
        self.trajectory.clear()


if __name__ == "__main__":
    os.environ["GRADIO_TEMP_DIR"] = "/net/acadia4a/data/zkhan/gradio-temp-dir"
    ray.init(address="auto")
    prompter = InsertQueryHerePrompter("prompts/2024-04-28-backticks_kvqa.xml")
    xml_renderer = SimpleXmlRenderer()
    config = OmegaConf.load("configs/eval_prototype.yaml") 

    agent = MistralVllmMReactAgentWithPinnedEngine(   # fucai check what is this
        prompter=prompter,
        observation_renderer=xml_renderer,
        vllm_engine_kwargs={
            "tensor_parallel_size": 1,
            "worker_use_ray": True,
        },
    )

    remote_environment_cls = ray.remote(JupyterVisualProgrammingEnvironment)
    environment = remote_environment_cls.remote(
        gateway_ip_addr=config.jupyter_kernel_gateway.ip_address,  # type: ignore
        gateway_port=config.jupyter_kernel_gateway.port,  # type: ignore
        module_specs=module_specs,
    )
    environment.reset.remote(image_path="example_data/fruits.png")  # type: ignore
    agent.reset("How many bananas are in the image?")

    gradio_address = get_external_address()

    action_queue: list[str] = []

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    value="/home/mai/zkhan/3-pale-giant/example_data/fruits.png",
                    type="filepath",
                )
                user_task_input = gr.Textbox(
                    value="How many bananas are in the image?",
                    visible=True,
                    label="Task Instruction",
                )
                user_or_env_message_input = gr.Textbox(
                    value="How many bananas are in the image?", visible=False
                )
                reset_btn = gr.Button("Reset Agent")
            with gr.Column():
                chatbot_view = gr.Chatbot(
                    sanitize_html=False, layout="bubble", label="Trajectory"
                )
                step_btn = gr.Button("Step")
                agent_action_view = gr.Markdown(label="Possible Action")
                see_action_btn = gr.Button("See Possible Action")
                accept_action_btn = gr.Button("Accept Action")

        def reset_all(query: str, image_path: str) -> tuple[str, list[tuple[str, str]]]:
            environment.reset.remote(image_path=image_path)  # type: ignore
            agent.reset(query=query)
            # Return an empty list to reset the chat history, and the user's
            # query as the first message to be displayed in the chatbot.
            return query, []

        def respond(
            user_or_env_message: str, chat_history: list[tuple[str, str]]
        ) -> tuple[str, list[tuple[str, str]]]:
            # TODO: The placement of the agent.update prevents us from easily applying
            # a user intervention. One possibility is to apply a user intervention after
            # as follows:
            # 1. agent.act() -> environment.step() -> agent.update()
            # 2. agent.act() -> user intervention -> agent.update()
            # But we cannot do this with the current setup because the current
            # implementation of this function does
            # 1. agent.act()
            # 2. environment.step() -> agent.update()
            # 3. <return control flow to the user>
            # 4. We need to see the next action before we can intervene.
            # So maybe we can just add a "see potential next action button?",
            # and if we don't like the action we can intervene before the action
            # is sent to the environment?

            # This will be the agent's response: the outcome of calling
            # agent.act()
            action = agent.act()
            # We call environment.step() with the agent's response to get the
            # next observation.
            observation = cast(Observation, ray.get(environment.step.remote(action=action)))  # type: ignore
            # We call agent.update() with the new observation to update the
            # agent's internal state.
            agent.update(experience=(action, observation))

            # We add the agent's response to the chat history. This is IGNORED
            # by the agent and is only used for visualization purposes.
            try:
                action_type, action_value = ray.get(
                    environment.parse_action.remote(action)  # type: ignore
                )
            except ValueError:
                rendered_action = action
            else:
                rendered_action = render_agent_action(action_type, action_value)

            chat_history.append((user_or_env_message, rendered_action))

            # We then format the observation for display in the chat interface.
            formatted_next_message = format_env_observation_for_display(observation)
            return formatted_next_message, chat_history

        def do_interaction_step(
            action: str,
            user_or_env_message: str,
            chat_history: list[tuple[str, str]],
        ):
            observation = cast(
                Observation, ray.get(environment.step.remote(action=action))  # type: ignore
            )
            agent.update(experience=(action, observation))
            try:
                action_type, action_value = ray.get(
                    environment.parse_action.remote(action)  # type: ignore
                )
            except ValueError:
                rendered_action = action
            else:
                rendered_action = render_agent_action(action_type, action_value)

            chat_history.append((user_or_env_message, rendered_action))
            formatted_next_message = format_env_observation_for_display(observation)
            return formatted_next_message, chat_history

        def do_interaction_step_from_queued_action(
            user_or_env_message: str,
            chat_history: list[tuple[str, str]],
        ):
            action = action_queue.pop()
            return do_interaction_step(action, user_or_env_message, chat_history)

        def view_possible_action():
            action = agent.act()
            action_queue.append(action)
            action_type, action_value = ray.get(
                environment.parse_action.remote(action)  # type: ignore
            )
            rendered_action = render_agent_action(action_type, action_value)
            return rendered_action

        step_btn.click(
            respond,
            [user_or_env_message_input, chatbot_view],
            [user_or_env_message_input, chatbot_view],
        )
        reset_btn.click(
            reset_all,
            [user_task_input, image_input],
            [user_or_env_message_input, chatbot_view],
        ).then(
            respond,
            [user_or_env_message_input, chatbot_view],
            [user_or_env_message_input, chatbot_view],
        )

        see_action_btn.click(
            view_possible_action,
            [],
            [agent_action_view],
        )

        accept_action_btn.click(
            do_interaction_step_from_queued_action,
            [user_or_env_message_input, chatbot_view],
            [user_or_env_message_input, chatbot_view],
        )

        demo.launch(
            server_name=gradio_address,
            server_port=GRADIO_SERVER_PORT,
            allowed_paths=["/home/mai/zkhan/3-pale-giant/example_data/"],
        )
