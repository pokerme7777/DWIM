import atexit
from dataclasses import dataclass
import os
from omegaconf import OmegaConf, DictConfig
import hydra
import sys
from loguru import logger
from pathlib import Path
import logging
import ray
from ray.util import ActorPool
from src.pale_giant_utils import JsonlIoHandler, InterceptHandler, ResumableIterator
from tqdm import tqdm
from typing import Optional
import json
from neurips_prototyping.new_worker import AgentEpisode
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy



@dataclass
class ExperimentState:
    remaining_iterations: ResumableIterator
    records_from_previous_run: list[dict]


def build_experiment_state(config, read_all_records_fn) -> ExperimentState:
    logger.info("Building dataset from _target_:{}", config.dataset._target_)
    dataset = hydra.utils.instantiate(config.dataset)
    logger.info("Built dataset with {} records", len(dataset))

    # This is used to ensure idempotence. If we set config.restart=true
    # and reuse the same config, the run will restart from where it last left
    # off. If the run is finished, it will do nothing and exist.
    if config.restart:
        logger.info("Restarting from previous run")
        try:
            records_from_previous_run = read_all_records_fn()
        except FileNotFoundError:
            logger.info("config.restart was set, but no previous run found")
            records_from_previous_run = []
        else:
            logger.info(
                "Read {} records from previous run", len(records_from_previous_run)
            )
    else:
        logger.info("Not restarting from previous run")
        records_from_previous_run = []

    resumable_iterator = ResumableIterator(
        dataset,
        config.iterations_per_record,
        records_from_previous_run,
    )
    return ExperimentState(
        remaining_iterations=resumable_iterator,
        records_from_previous_run=records_from_previous_run,
    )


def build_ray_iterator(config, worker_cls, dataset_iterator):
    ray.init()

    remote_worker_cls = ray.remote(worker_cls)
    workers = [
        remote_worker_cls.remote(
            environment_config=config.agentic_task_runner.environment,
            agent_config=config.agentic_task_runner.agent,
            max_steps=config.agentic_task_runner.max_steps,
            caption_first=config.caption_first,
            instructed_mode=config.instructed_mode,
            use_refered_answer=config.use_refered_answer,
            NF_flag=config.NF_flag,
            worker_id=f"{config.experiment_namespace}/worker_{_}",
        )
        for _ in range(config.ray.num_workers)  # type: ignore[attr-defined]
    ]
    worker_pool = ActorPool(workers)
    ray_iterator = worker_pool.map_unordered(
        lambda worker, record: worker.run.remote(record=record), dataset_iterator  # type: ignore
    )  # important fucai

    return workers, ray_iterator


def compute_metrics(
    config,
    records,
    records_from_previous_run: Optional[list[dict]] = None,
):
    metrics_calculator = hydra.utils.instantiate(config.metrics.calculator)
    metrics = metrics_calculator(
        records + (records_from_previous_run or []),
    )
    logger.info("Metrics: {}", OmegaConf.to_yaml(metrics))

    return metrics


def collect_results(
    config, ray_iterator, dataset_iterator, result_save_fn
) -> list[dict]:
    metrics_formatter = hydra.utils.instantiate(config.metrics.running_formatter)
    records = []
    enumerator = tqdm(enumerate(ray_iterator), total=len(dataset_iterator))
    for idx, result in enumerator:
        if result is None:
            continue
        maybe_agentic_inference_step_output, instance = result

        if maybe_agentic_inference_step_output is None:
            # We could not find the image for this record.
            continue
        else:
            agentic_inference_step_output = maybe_agentic_inference_step_output

        # Use a dict to store everything so we can ignore ones
        # we have already obtained an answer for.
        record = {
            # "program": agentic_inference_step_output.program,
            "result": agentic_inference_step_output.result,
            "label": instance["label"],
            "failure_mode": agentic_inference_step_output.failure_mode.value,
            "question": instance["question"],
            "caption":instance["caption"],
            "question_id": instance["question_id"],
            "tokens": -1,
            "image_id": instance["image_id"],
            "question_type": instance.get("question_type", None),
            "trajectory": [
                _.model_dump() for _ in agentic_inference_step_output.trajectory
            ],
        }

        metrics_formatter.append(record)

        if idx % 10 == 0 and idx > 0:
            metrics_formatter.log()

        result_save_fn(record)
        records.append(record)

    # return records[:600]
    return records


def setup_logging():
    # Set up Python's default logging.
    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    # Delete the default logger and add one that logs at INFO.
    logger.remove()
    logger.add(sys.stdout, level="INFO", colorize=True)
    os.environ["LOGURU_LEVEL"] = "INFO"


class ExperimentIOProvider:
    def __init__(self, output_folder, make_folder_fn=os.makedirs):
        make_folder_fn(output_folder, exist_ok=True)
        self.output_folder = Path(output_folder)
        self.jsonl_io_handler = JsonlIoHandler(
            str(self.output_folder / "records.jsonl")
        )

    def _save_json(self, data, filename):
        with open(self.output_folder / filename, "w") as f:
            json.dump(data, f)

    def save_config(self, config):
        OmegaConf.save(config, self.output_folder / "config.yaml")

    def save_record(self, record: dict):
        self.jsonl_io_handler.append_dict(record)

    def save_metrics(self, metrics: dict):
        self._save_json(metrics, "metrics.json")

    def read_all_records(self) -> list[dict]:
        return self.jsonl_io_handler.read_all()

    @classmethod
    def from_config(cls, config):
        return cls(output_folder=config.output_folder)


def main(config: DictConfig):
    setup_logging()

    logger.info("\n{}", OmegaConf.to_yaml(config))

    experiment_io_provider = ExperimentIOProvider.from_config(config)
    experiment_io_provider.save_config(config)

    experiment_state = build_experiment_state(
        config,
        experiment_io_provider.read_all_records,
    )

    worker_cls = hydra.utils.get_class(config.agentic_task_runner._target_)
    ray_workers, ray_iterator = build_ray_iterator(
        config, worker_cls, experiment_state.remaining_iterations
    ) # important fucai

    atexit.register(lambda: [_.close.remote() for _ in ray_workers])
    records = collect_results(
        config,
        ray_iterator,
        experiment_state.remaining_iterations,
        experiment_io_provider.save_record,
    )

    metrics = compute_metrics(
        config,
        records,
        experiment_state.records_from_previous_run,
    )
    experiment_io_provider.save_metrics(metrics)


if __name__ == "__main__":
    overrides = sys.argv[1:]
    hydra.initialize(version_base=None, config_path="configs")
    config = hydra.compose(config_name="eval_prototype", overrides=overrides)
    OmegaConf.resolve(config)
    assert isinstance(config, DictConfig)

    main(config)
