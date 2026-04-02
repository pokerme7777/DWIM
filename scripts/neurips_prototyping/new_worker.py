import sys

import hydra
import ray
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field
from vllm.utils import get_ip

from src.instrumentation import FailureMode

from .environment import (
    Observation,
    ObservationV1,
    retrieve_final_result_from_trajectory,
)
from .jupyter import JupyterKernelGatewayWrapper


class Experience(BaseModel):
    action: str
    observation: Observation = Field(..., discriminator="observation_type")


# This class is included as a compatability shim so we can load trajectories
# from older code, which did not use a discriminator field in their observations.
# Because they did not use a discriminator field in their observations, Pydantic cannot
# tell which observation to deserialize to. This class has a workaround that uses
# a callable discriminator function to determine which observation to deserialize to.
# Afterwards, you can convert this class to the newer Experience class.
class ExperienceV1(BaseModel):
    action: str
    observation: ObservationV1

    def to_experience(self) -> Experience:
        return Experience(action=self.action, observation=self.observation)


class AgentEpisode(BaseModel):
    trajectory: list[Experience] = []
    result: str = "MISSING_RESULT"
    program: str = "MISSING_PROGRAM"
    failure_mode: FailureMode = FailureMode.NO_FAILURE

    def model_post_init(self, __context):
        observations = [exp.observation for exp in self.trajectory]
        self.result = retrieve_final_result_from_trajectory(observations)


# class StubAgent:
#     def __init__(self):
#         self.num_invocations = 0

#     def act(self):
#         if self.num_invocations == 5:
#             return """<code>ImagePatch(image).find("banana")</code>"""
#         return f"""<code>print("num invocations={self.num_invocations}")</code>"""

#     def update(self, experience):
#         self.num_invocations += 1

#     def reset(self, query: str):
#         self.num_invocations = 0

# key for running agent. fucai
class AgenticTaskRunner:
    def __init__(
        self,
        environment_config: DictConfig,
        agent_config: DictConfig,
        max_steps: int = 10, 
        caption_first : int = 0, # 0 is Flase, 1 is captioning first using VLM, 2 is using preproced caption.
        use_refered_answer : int = 0, # 0 is flase, 1 mean using refered answer for training.
        instructed_mode : int = 0, # 0 is false, 1 is using instruction tuning mode
        NF_flag: int = 0, # new feedback
        worker_id: str = "worker_namespace",
    ):
        self.max_steps = max_steps
        self.worker_id = worker_id
        self.agent_config = agent_config
        self.environment_config = environment_config
        self.caption_first = caption_first
        self.instructed_mode = instructed_mode
        self.use_refered_answer = use_refered_answer
        self.NF_flag = NF_flag

        # We do not call setup() here because there are complications with
        # Ray's serialization when the __init__ is doing a bunch of complicated
        # stuff. Instead, we call `setup` lazily in the `run` method when the
        # task runner has been instantiated and distributed (if necessary).
        self._setup_complete = False
        self.environment = None
        self.agent = None

    @classmethod
    def build_from_config(
        cls, config: DictConfig, worker_id: str
    ) -> "AgenticTaskRunner":
        return cls(
            environment_config=config.agentic_task_runner.environment,
            agent_config=config.agentic_task_runner.agent,
            max_steps=config.agentic_task_runner.max_steps,
            caption_first=config.caption_first==1,
            worker_id=worker_id,
        )

    def setup(self):
        # IMPORTANT: Build the agent before you build the environment.
        # Ray messes up (I think) and gives the wrong GPU ID to the agent
        # otherwise.
        ip_address = get_ip()
        logger.info(
            "Setting up task runner for worker_id={} at ip={}",
            self.worker_id,
            ip_address,
        )
        self.agent = self.build_agent()
        self.environment = self.build_environment()
        self._setup_complete = True
        logger.info("Setup complete for worker_id={}", self.worker_id)

    def build_environment(self):
        # NOTE: Using `.call` here instead if `instantiate` is an
        # intentional decision because we want to keep the interface
        # of the environment clean and typed, but the `list[ModuleSpec]`
        # needed by the environment have to be constructed dynamically
        # from the serialized config; the environment has a method to
        # construct itself from such a serialized config.
        environment = hydra.utils.call(
            self.environment_config, namespace=self.worker_id , NF_flag=self.NF_flag,
        )
        return environment

    def build_agent(self):
        agent = hydra.utils.instantiate(self.agent_config)
        return agent

    def _run(self, record: dict) -> AgentEpisode:
        if not self._setup_complete:
            self.setup()

        if self.environment is None or self.agent is None:
            raise ValueError(
                "Environment and agent must not be None: environment={}, agent={}".format(
                    self.environment, self.agent
                )
            )

        # self.agent.reset(query=record["question"])
        self.environment.reset(image_path=record["image_id"])
        logger.info(
            "Beginning episode with for query={} and image_id={}",
            record["question"],
            record["image_id"],
        )

        trajectory = []
        # # fucai caption  
        if self.caption_first ==1: # need to generate caption by llava or VLM
            # action = """<code>patch = ImagePatch(image)\nprint(patch.complex_query("Could you please describe this image? The information should be more about {}"))</code>""".format(str(record["question"]))
            if 'sugar' in record['question_type'] or 'wino' in record['question_type']:
                action = """<code>patch = ImagePatch(image)\npatch.captioning("Provide a descriptive caption summarizing the main elements and action in this image.")</code>"""
            else:    
                action = """<code>patch = ImagePatch(image)\npatch.captioning("Could you please describe this image? The information should be more about {}")</code>""".format(str(record["question"]))
            # get caption
            caption_observation = self.environment.step(action).execution_result
        elif self.caption_first ==2: # using pre processing caption from text2vec or other"
            caption_observation = record["prost_captioning"]
        else: 
            caption_observation = 'None'

        # instruct refered answer
        if self.use_refered_answer ==1:
            refered_answer = record["label"]
        else:
            refered_answer = 'no refered answer'

        #reset the agent !!!
        if isinstance(refered_answer, str):
            self.agent.reset(query=record["question"], caption_observation=caption_observation, refered_answer=refered_answer)
        else:
            self.agent.reset(query=record["question"], caption_observation=caption_observation, refered_answer=str(refered_answer))
        
        #sace caption into record
        record["caption"] = caption_observation

        for step in range(self.max_steps):
            action = self.agent.act()
            observation = self.environment.step(action)

            # if instruction tuning, we need to have a "Step no:" in the begining of next line.
            if self.instructed_mode ==1 and not self.environment.is_done():
                add_step_flag = True
            else:
                add_step_flag = False
            
            # Becasue I do not have time to change other agent, I only change llama31_react_agent. So using try except.
            try:
                self.agent.update(experience=(action, observation), add_step_flag=add_step_flag, step_no=step+2)
            except:
                self.agent.update(experience=(action, observation))

            trajectory.append(Experience(action=action, observation=observation))

            if self.environment.is_done():
                logger.info("Trajectory complete at step={}", step)
                break
        else:
            logger.info("Trajectory incomplete after max_steps={}", self.max_steps)

        # # testing   fucai debuging
        # with open('/net/acadia7a/data/fkee/learning_agent/output.txt', 'w') as f:
        #     for item in self.agent.state:
        #         f.write(f"{item}\n")

        return AgentEpisode(trajectory=trajectory), record

    def run(self, record: dict) -> tuple[AgentEpisode, dict]:
        try:
            episode, record = self._run(record)
        except KeyboardInterrupt:
            raise
        except Exception:
            logger.opt(exception=True).error(
                "Exception while processing record={}", record["question_id"]
            )
            # TODO: I don't think this is the right error code. We probably
            # need to define a new error code for this.
            return (
                AgentEpisode(trajectory=[], failure_mode=FailureMode.PROGRAM_EXECUTION),
                record,
            )
        else:
            return episode, record

    def close(self):
        # When being invoked as a remote class, the `.environment` attribute
        # will appear to be missing in the destructor. We do nothing in this case.
        if not hasattr(self, "environment"):
            return
        if self.environment is not None:
            self.environment.close()
        self.environment = None

    def __del__(self):
        self.close()


if __name__ == "__main__":
    overrides = sys.argv[1:]
    hydra.initialize(version_base=None, config_path="../configs")
    config = hydra.compose(config_name="eval_prototype", overrides=overrides)
    OmegaConf.resolve(config)
    assert isinstance(config, DictConfig)

    ray.init(address="auto")
    with JupyterKernelGatewayWrapper(
        ip_address=config.jupyter_kernel_gateway.ip_address,
        port=config.jupyter_kernel_gateway.port,
    ) as gateway:
        agentic_task_runner = AgenticTaskRunner.build_from_config(
            config, worker_id="shmeegum"
        )
        agentic_task_runner.run(
            {
                "image_id": "example_data/fruits.png",
                "question": "How many bananas are present?",
            }
        )

