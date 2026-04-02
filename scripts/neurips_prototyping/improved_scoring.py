"""
This file implements more advanced rule-based scoring for VQA tasks. 
We developed it for GQA originally — check notebooks 095 and 098 for more details.
The motivation is that exact match based scoring is harsh. To avoid penalizing the agent
overmuch, we have to carefully parse the responses from the agent.
"""

import ast
import re
import time
from typing import Callable, Generator, Literal, Union

import numpy as np
import pandas as pd
import ray
import rich
from loguru import logger
from pydantic import BaseModel
from ray.util.placement_group import (
    placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, LLMEngine, RequestOutput, SamplingParams

from src.instrumentation import calculate_vqa_v2_exact_match_score

ACTOR_NAMESPACE = "improved_scoring"


class JudgeFunctionInput(BaseModel):
    scorable_value: str
    ground_truth: Union[str, list[str]]
    question: str
    image_path: str


def stub_judge_function(scorables: list[JudgeFunctionInput]) -> list[float]:
    return [-3.14] * len(scorables)


def judge_as_wrong(scorables: list[JudgeFunctionInput]) -> list[float]:
    return [0.0] * len(scorables)


class Scorer:
    def __init__(
        self,
        records,
        judge_fn: Callable[[list[JudgeFunctionInput]], list[float]],
        exact_match_fn: Callable[
            [str, str], float
        ] = calculate_vqa_v2_exact_match_score,
    ):
        self.records = records
        self.judge_fn = judge_fn
        self.exact_match_fn = exact_match_fn

    def do_judging(self) -> None:
        scoring_tasks: list[
            tuple[
                Union[
                    Literal["use_exact_match"],
                    Literal["use_judge_fn"],
                    Literal["force_mark_invalid"],
                ],
                str,
                dict,
            ]
        ] = []

        for record in self.records:
            scorer, scorable_value = self.select_scoring_strategy_for_record(record)
            scoring_tasks.append((scorer, scorable_value, record))

        judge_as_invalid = [
            (scorable_value, record)
            for (scorer, scorable_value, record) in scoring_tasks
            if scorer == "force_mark_invalid"
        ]

        judge_with_exact_matches = [
            (scorable_value, record)
            for (scorer, scorable_value, record) in scoring_tasks
            if scorer == "use_exact_match"
        ]

        judge_with_custom_fn = [
            (scorable_value, record)
            for (scorer, scorable_value, record) in scoring_tasks
            if scorer == "use_judge_fn"
        ]

        # This will mutate the records in place so they contain the scores.
        # TODO: Fix this pattern to be more functional.
        self.judge_with_exact_matches(judge_with_exact_matches)
        self.judge_with_judge_fn(judge_with_custom_fn)
        self.skip_judging_and_mark_invalid(judge_as_invalid)

    def skip_judging_and_mark_invalid(self, scorables: list[tuple[str, dict]]) -> None:
        for _, record in scorables:
            record["score"] = 0
            record["scorable_value"] = "NOT_USED"
            record["scoring_strategy"] = "force_mark_invalid"

    def judge_with_exact_matches(self, scorables: list[tuple[str, dict]]) -> None:
        for scorable_value, record in scorables:
            ground_truth = record["label"]
            score = self.exact_match_fn(scorable_value, ground_truth)
            record["score"] = score
            record["scorable_value"] = scorable_value
            record["scoring_strategy"] = "exact_match"

    def judge_with_judge_fn(self, scorables: list[tuple[str, dict]]) -> None:
        judge_fn_inputs = [
            JudgeFunctionInput(
                scorable_value=str(scorable_value),
                ground_truth=record["label"],
                question=record["question"],
                image_path=record["image_id"],
            )
            for scorable_value, record in scorables
        ]
        scores = self.judge_fn(judge_fn_inputs)
        for score, (scorable_value, record) in zip(scores, scorables):
            record["score"] = score
            record["scorable_value"] = scorable_value
            record["scoring_strategy"] = "judge_fn"

    def select_scoring_strategy_for_record(self, record: dict) -> tuple[
        Union[
            Literal["use_exact_match"],
            Literal["use_judge_fn"],
            Literal["force_mark_invalid"],
        ],
        str,
    ]:
        prediction = record["result"]
        ground_truth = record["label"]

        # We first invalidate predictions that are not scorable.
        if self.check_prediction_valid_for_scoring(prediction):
            return "force_mark_invalid", prediction

        # Try our hardest to normalize the prediction to a boolean yes
        # or no answer.
        maybe_boolean_prediction = self.try_normalize_to_yesno(prediction)
        if maybe_boolean_prediction in ("yes", "no"):
            return "use_exact_match", maybe_boolean_prediction

        # The prediction is not a yes or no, but it could still
        # pass an exact match test as a string. For example, the answer
        # could be "hippo" and the prediction could be "hippo".
        normalized_prediction = self.try_make_literal(prediction)
        if self.exact_match_fn(normalized_prediction, ground_truth) > 0:
            return "use_exact_match", normalized_prediction

        # This is not an exact match, but it doesn't mean the prediction is wrong.
        # We have to use a custom judge function to determine if the prediction is correct.
        return "use_judge_fn", normalized_prediction

    def check_prediction_valid_for_scoring(self, prediction: str) -> bool:
        return prediction in (
            "NO_CODE_OBSERVATIONS",
            "FINAL_ANSWER_NOT_FOUND",
            "CANNOT_STRINGIFY_PROGRAM_STATE",
            "CANNOT_DESERIALIZE_PROGRAM_STATE",
        )

    def try_make_literal(self, prediction: str) -> str:
        try:
            return ast.literal_eval(prediction)
        except (SyntaxError, ValueError):
            return prediction

    def try_normalize_to_yesno(
        self, prediction: str
    ) -> Union[Literal["yes"], Literal["no"], Literal["not_yesno"]]:
        maybe_truthable_literal = self.try_parse_as_truthable_literal(prediction)
        if maybe_truthable_literal is not None:
            return maybe_truthable_literal

        maybe_yesno = self.try_parse_string_as_yesno(prediction)
        if maybe_yesno in ("yes", "no"):
            return maybe_yesno

        return "not_yesno"

    def try_parse_string_as_yesno(
        self,
        answer: str,
    ) -> Union[Literal["yes"], Literal["no"], Literal["not_yesno"]]:
        # Check if there is a literal true or false in the answer
        if "true" in answer.lower() or "false" in answer.lower():
            # Return yes if true is present in the answer
            return "yes" if "true" in answer.lower() else "no"
        # Check if there is a true or false surrounded by non word boundary characters
        elif re.match(r"\B(true|false)\B", answer, re.IGNORECASE):
            # Return yes if true is present in the answer anywhere
            return "yes" if re.search(r"\btrue\b", answer, re.IGNORECASE) else "no"
        elif answer.lower() == "none":
            return "no"
        elif re.match(r"^\[\]$", answer):
            return "no"
        elif re.match(r"^\[\s*(.+)\s*\]$", answer):
            return "yes"
        elif re.match(r"^\d+$", answer.strip()):
            return "yes" if int(answer.strip()) != 0 else "no"
        # Check if there is a literal yes or no in the answer
        elif re.search(r"\b(yes|no)\b", answer, re.IGNORECASE):
            # return yes if yes is present in the answer
            return "yes" if re.search(r"\byes\b", answer, re.IGNORECASE) else "no"
        else:
            return "not_yesno"

    def try_parse_as_truthable_literal(
        self, raw_answer: str
    ) -> Union[Literal["yes"], Literal["no"], None]:
        try:
            literal_answer = ast.literal_eval(raw_answer)
            match literal_answer:
                # We can parse None, booleans, lists, and ints as truthy or
                # falsy values fairly unambiguously.
                case bool():
                    return "yes" if literal_answer else "no"
                case list():
                    return "yes" if literal_answer else "no"
                case int():
                    return "yes" if literal_answer else "no"
                case None:
                    return "no"
                # A string _might_ be a yes or no answer, but it might not be.
                # We leave the parsing of strings to a dedicated function.
                case str():
                    return None
                case _:
                    return None

        except (SyntaxError, ValueError):
            return None


class SanityCheckScoredRecords:
    def __init__(self, records: list[dict]):
        self.records = records

    def calculate_exact_match_score(self):
        used_exact_match = [
            _ for _ in self.records if _["scoring_strategy"] == "exact_match"
        ]
        exact_match_correct = [_ for _ in used_exact_match if _["score"] > 0]

        accuracy_relative = len(exact_match_correct) / len(used_exact_match)
        accuracy_absolute = len(exact_match_correct) / len(self.records)
        return {
            "accuracy_relative": accuracy_relative,
            "accuracy_absolute": accuracy_absolute,
        }

    def calculate_percentage_not_scorable(self):
        df = pd.DataFrame(self.records, columns=["failure_mode", "result"])
        not_scorable = df[
            df["failure_mode"].eq("program_execution")
            | (df["result"].isin({"CANNOT_STRINGIFY_PROGRAM_STATE"}))
        ]
        # Disentangle the number which failed due to program execution, ones which failed due
        # to stringification, and ones which failed due to both.
        program_execution_failure = df[df["failure_mode"].eq("program_execution")]
        stringification_failure = df[df["result"].eq("CANNOT_STRINGIFY_PROGRAM_STATE")]
        no_code_observation_failure = df[df["result"].eq("NO_CODE_OBSERVATIONS")]
        no_final_result_failure = df[df["result"].eq("FINAL_ANSWER_NOT_FOUND")]
        not_able_to_deserialize_program_state = df[
            df["result"].eq("CANNOT_DESERIALIZE_PROGRAM_STATE")
        ]

        return {
            "not_scorable": len(not_scorable),
            "program_execution_failure": len(program_execution_failure),
            "stringification_failure": len(stringification_failure),
            "no_code_observation_failure": len(no_code_observation_failure),
            "no_final_result_failure": len(no_final_result_failure),
            "not_able_to_deserialize_program_state": len(
                not_able_to_deserialize_program_state
            ),
        }

    def calculate_scoring_strategy_breakdown(self):
        df = pd.DataFrame(self.records, columns=["scoring_strategy", "score"])
        strategy_breakdown = df.groupby("scoring_strategy").agg(
            {"score": ["mean", "count"]}
        )
        # Also calculate the overall score and add it to the breakdown.
        overall_score = df["score"].mean()
        overall_count = len(df)
        strategy_breakdown.loc["overall"] = (overall_score, overall_count)
        return strategy_breakdown

    def calculate_unintentionally_correct_answers(self):
        # Unintentionally correct answers are those for which the
        # trajectory failed to execute, but the hardcoded default answer
        # was correct.
        df = pd.DataFrame(
            self.records, columns=["label", "result", "failure_mode", "score"]
        )
        unintentionally_correct = df[
            df["failure_mode"].eq("program_execution") & df["score"].gt(0)
        ]
        return {
            "unintentionally_correct": len(unintentionally_correct),
            "percentage_unintentionally_correct": len(unintentionally_correct)
            / len(df),
        }

    def calculate_marked_invalid_breakdown(self):
        df = pd.DataFrame(self.records, columns=["scoring_strategy", "result"])
        marked_invalid = df[df["scoring_strategy"].eq("force_mark_invalid")]
        reasons = marked_invalid["result"].value_counts()

        return {
            "marked_invalid": len(marked_invalid),
            "reasons": reasons,
        }

    def calculate_details_of_execution_failures(self):
        df = pd.DataFrame(self.records, columns=["result", "failure_mode"])
        execution_failures = df[df["failure_mode"].eq("program_execution")]
        return {
            "execution_failures": len(execution_failures),
            "reasons": execution_failures["result"].value_counts(),
        }

    def calculate_breakdown_of_exact_match_scores(self):
        df = pd.DataFrame(
            self.records,
            columns=[
                "score",
                "scoring_strategy",
                "label",
                "question",
                "scorable_value",
            ],
        )
        used_exact_match = df[df["scoring_strategy"].eq("exact_match")]
        was_yes = used_exact_match[used_exact_match["label"].eq("yes")]
        was_no = used_exact_match[used_exact_match["label"].eq("no")]
        was_not_boolean = used_exact_match[
            ~used_exact_match["label"].isin({"yes", "no"})
        ]

        # For each of these, calculate how many were correct, and how many were incorrect.
        yes_correct = was_yes[was_yes["score"].gt(0)]
        no_correct = was_no[was_no["score"].gt(0)]
        not_boolean_correct = was_not_boolean[was_not_boolean["score"].gt(0)]

        # Calculate accuracies for each of these.
        if len(was_yes) == 0:
            yes_accuracy = np.nan
        else:
            yes_accuracy = len(yes_correct) / len(was_yes)

        if len(was_no) == 0:
            no_accuracy = np.nan
        else:
            no_accuracy = len(no_correct) / len(was_no)

        if len(was_not_boolean) == 0:
            not_boolean_accuracy = np.nan
        else:
            not_boolean_accuracy = len(not_boolean_correct) / len(was_not_boolean)

        # Get labels and normalized answers for not_boolean incorrects.
        not_boolean_incorrect = was_not_boolean[was_not_boolean["score"].eq(0)]

        return {
            "yes_accuracy": yes_accuracy,
            "no_accuracy": no_accuracy,
            "not_boolean_accuracy": not_boolean_accuracy,
            "num_ground_truth_yes": len(was_yes),
            "num_ground_truth_no": len(was_no),
            "num_ground_truth_not_boolean": len(was_not_boolean),
            "not_boolean_incorrect": not_boolean_incorrect,
        }

    def calculate_summary_statistics_of_correct_trajectories(self) -> dict:
        correct_trajectories = [_ for _ in self.records if _["score"] > 0]
        length_distribution = [len(_["trajectory"]) for _ in correct_trajectories]

        # Calculate some summary statistics.
        mean_length = np.mean(length_distribution)
        median_length = np.mean(length_distribution)
        percentiles = np.percentile(length_distribution, [25, 50, 75])
        # Zip up the percentiles so they're easier to display.
        percentiles_for_display = dict(zip([25, 50, 75], percentiles))
        max_length = np.max(length_distribution)
        min_length = np.min(length_distribution)

        return {
            "mean_length": mean_length,
            "median_length": median_length,
            "percentiles": percentiles_for_display,
            "max_length": max_length,
            "min_length": min_length,
        }

    def __call__(self) -> None:
        rich.print(self.calculate_exact_match_score())
        rich.print(self.calculate_percentage_not_scorable())
        rich.print(self.calculate_scoring_strategy_breakdown())
        rich.print(self.calculate_unintentionally_correct_answers())
        rich.print(self.calculate_marked_invalid_breakdown())
        rich.print(self.calculate_details_of_execution_failures())
        rich.print(self.calculate_breakdown_of_exact_match_scores())
        rich.print(self.calculate_summary_statistics_of_correct_trajectories())


class Mistral7BJudge:
    class Ray:
        actor_namespace = ACTOR_NAMESPACE
        actor_name = "Mistral7BJudge_{slug}"

    def __init__(self, is_remote: bool = False) -> None:
        self.human_judgements = pd.read_csv(
            "notebooks/artifacts/nb095_human_judgements_of_answer_correctness.tsv",
            sep="\t",
        )
        self.instruction = """You are an expert and careful judge. You will be given a question, a contestant's answer, and a correct reference answer. Your job is to check the contestant's answer against the provided reference answer and determine whether it is similar enough to be considered correct in the context of the question.
        Always begin your response with [judgement=pass|fail] to indicate whether the contestant's answer is correct or not."""
        self.example_template = "Question: {question}\nReference Answer: {ground_truth}\nContestant's Answer: {scorable_value}\n"
        self.system_message = {"role": "user", "content": self.instruction}
        self.assistant_acknowledgement = {
            "role": "assistant",
            "content": "I understand the instructions.",
        }

        self.in_context_examples: list[dict[str, str]] = []
        for record in self.human_judgements.itertuples():
            self.in_context_examples.append(
                {
                    "role": "user",
                    "content": self.example_template.format(
                        question=record.question,
                        ground_truth=record.ground_truth,
                        scorable_value=record.scorable_value,
                    ),
                }
            )
            self.in_context_examples.append(
                {
                    "role": "assistant",
                    "content": f"[judgement={'pass' if int(record.human_judgement) == 1 else 'fail'}]",
                }  # type: ignore
            )
        if is_remote:
            self.llm = LLM(
                model="mistralai/Mistral-7B-Instruct-v0.2", worker_use_ray=True
            )
        else:
            self.llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2"
        )
        self.sampling_params = SamplingParams(
            max_tokens=30, temperature=0.0, top_p=0.95
        )

    @classmethod
    def instantiate_as_remote_actor(cls):
        placement_group_specs = [{"GPU": 1.0}]
        # Ensure one CPU is available for the driver process.
        placement_group_specs += [{"CPU": 1.0}]
        pg = placement_group(placement_group_specs, strategy="STRICT_PACK")
        # Block until the placement group is ready.
        ray.get(pg.ready(), timeout=10)
        logger.info("Placement group ready: {}", pg)

        # Note: we cannot make this detached, because the lifetime of
        # this actor is dependent on the placement group, which will
        # be removed once the driver script exits. You do not want to
        # make the placement group detached because that will make
        # resource management and cleanup very annoying.

        namespace = cls.Ray.actor_namespace
        # Give it a unique name to prevent collisions
        # and possibly tearing down other actors.
        name = cls.Ray.actor_name.format(slug=str(time.time()))

        logger.info(
            "Instantiating remote judge at {namespace}/{name}",
            namespace=namespace,
            name=name,
        )
        constructor = ray.remote(cls).options(
            num_cpus=1,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group_capture_child_tasks=True,
                placement_group=pg,
            ),
            namespace=namespace,
            name=name,
        )

        return constructor.remote(is_remote=True)

    @classmethod
    def get_actor_handle(cls):
        actor_handle = ray.get_actor(
            name=cls.Ray.actor_name, namespace=cls.Ray.actor_namespace
        )
        return actor_handle

    def prepare_conversation_state_for_judging(
        self, question: str, ground_truth: str, scorable_value: str
    ) -> list[dict[str, str]]:
        conversation = (
            [self.system_message, self.assistant_acknowledgement]
            + self.in_context_examples
            + [self.system_message, self.assistant_acknowledgement]
        )
        conversation.append(
            {
                "role": "user",
                "content": self.example_template.format(
                    question=question,
                    ground_truth=ground_truth,
                    scorable_value=scorable_value,
                ),
            }
        )
        return conversation

    @staticmethod
    def process_requests(
        engine: LLMEngine, prompts: list[tuple[str, SamplingParams]]
    ) -> Generator[RequestOutput, None, None]:
        """Continuously process a list of prompts and handle the outputs."""
        request_id = 0

        while prompts or engine.has_unfinished_requests():
            if prompts:
                prompt, sampling_params = prompts.pop(0)
                engine.add_request(str(request_id), prompt, sampling_params)
                request_id += 1

            request_outputs: list[RequestOutput] = engine.step()

            for request_output in request_outputs:
                if request_output.finished:
                    yield request_output

    @staticmethod
    def parse_judgement_from_llm(
        response: str,
    ) -> Union[bool, Literal["failed_to_judge"]]:
        # Do a regex to check for judgement=pass or judgement=fail
        # and capture the value with a named group.
        # Ignore the brackets because sometimes they aren't
        # produced, but we don't care.
        pattern = re.compile(r"judgement=(?P<judgement>pass|fail)")
        match = pattern.search(response)
        if match:
            return match.group("judgement") == "pass"

        return "failed_to_judge"

    def __call__(self, items_to_judge: list[JudgeFunctionInput]) -> list[float]:
        prompts: list[tuple[str, SamplingParams]] = []
        for to_judge in items_to_judge:
            if not isinstance(to_judge.ground_truth, str):
                raise NotImplementedError(
                    "Scoring for multiple ground truths not implemented."
                )
            conversation = self.prepare_conversation_state_for_judging(
                question=to_judge.question,
                ground_truth=to_judge.ground_truth,
                scorable_value=to_judge.scorable_value,
            )
            prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False)
            prompts.append((prompt, self.sampling_params))

        responses = [
            _
            for _ in tqdm(
                self.process_requests(self.llm.llm_engine, prompts), total=len(prompts)
            )
        ]

        # Sort the responses by request_id to match them with the input.
        responses = sorted(responses, key=lambda _: int(_.request_id))

        judgments: list[float] = []
        for idx, (to_judge, response) in enumerate(zip(items_to_judge, responses)):
            assert int(response.request_id) == idx
            response_text = response.outputs[0].text
            judgement = self.parse_judgement_from_llm(response_text)
            if judgement == "failed_to_judge":
                logger.warning(
                    f"Failed to judge scorable_value={to_judge.scorable_value} for question={to_judge.question}, scoring as 0.0, judge_response={response_text}"
                )
                judgments.append(0.0)
            else:
                judgments.append(1.0 if judgement else 0.0)

        return judgments


if __name__ == "__main__":
    from src.pale_giant_utils import JsonlIoHandler

    def records_factory(which_records: str = "control") -> list[dict]:
        if which_records == "control":
            records_path = (
                "cache/111_eval_prototyp_gqa_improved_prompt_8x7b_gqa/records.jsonl"
            )
        elif which_records == "treatment":
            records_path = "cache/118_eval_mistral_8x7b_gqa_retrieved_ice/records.jsonl"
        else:
            raise ValueError(f"Unknown records type: {which_records}")

        io_handler = JsonlIoHandler(records_path)
        return io_handler.read_all()

    def do_scoring(which_records: str = "control"):
        records = records_factory(which_records)
        scorer = Scorer(records=records, judge_fn=stub_judge_function)
        scorer.do_judging()
        sanity_checker = SanityCheckScoredRecords(scorer.records)
        sanity_checker()
        return sanity_checker

    control_scoring_sanity_checker = do_scoring("control")

    treatment_scoring_sanity_checker = do_scoring("treatment")

    mistral_judge_fn = Mistral7BJudge()

    def do_scoring_with_mistral_judge(which_records: str = "control"):
        records = records_factory(which_records)
        scorer = Scorer(records=records, judge_fn=mistral_judge_fn)
        scorer.do_judging()
        sanity_checker = SanityCheckScoredRecords(scorer.records)
        sanity_checker()
        return sanity_checker

    control_scoring_sanity_checker = do_scoring_with_mistral_judge("control")
    treatment_scoring_sanity_checker = do_scoring_with_mistral_judge("treatment")
