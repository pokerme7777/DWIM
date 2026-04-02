import enum
from types import MethodDescriptorType
from typing import Any, Callable, Iterable, Optional, NewType, TypeVar, Type
import numpy as np
from collections.abc import Sequence
import pandas as pd
from typing import cast, Union
from loguru import logger
from scipy.stats import beta
import math
from collections import Counter, defaultdict
from numbers import Number
from tqdm.contrib.logging import logging_redirect_tqdm
import regex
import string
from functools import wraps, update_wrapper
import inspect
from contextlib import contextmanager
from pydantic import BaseModel
import ast


def is_numeric(value) -> bool:
    """Check if a value is numeric."""
    return isinstance(value, Number)


class FailureMode(enum.Enum):
    PROGRAM_SYNTHESIS = "program_synthesis"
    PROGRAM_PARSING = "program_parsing"
    PROGRAM_COMPILATION = "program_compilation"
    PROGRAM_NAME_ERROR = "program_name_error"
    PROGRAM_EXECUTION = "program_execution"
    NO_FAILURE = "no_failure"


def calculate_percentage_of_failure_mode(
    records: list[dict[str, Any]], failure_mode: FailureMode
) -> float:
    total = len(records)
    count = 0
    for record in records:
        if record["failure_mode"] == failure_mode.value:
            count += 1
    return count / total


def calculate_percentage_of_all_failure_modes(
    records: list[dict[str, Any]]
) -> dict[FailureMode, float]:
    total = len(records)
    counts = {failure_mode: 0 for failure_mode in FailureMode}
    for record in records:
        counts[FailureMode(record["failure_mode"])] += 1
    return {failure_mode: count / total for failure_mode, count in counts.items()}


# This is taken from the KAT code.
def normalize_answer(s: str):
    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def calculate_normalized_exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def calculate_vqa_v2_exact_match_score(
    prediction: str, ground_truths: Union[Sequence[str], str]
) -> float:
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    elif isinstance(ground_truths, int):
        ground_truths = [str(ground_truths)]

    if not isinstance(prediction, str):
        # This can occasionally happen in the case that
        # the model produces a list of predictions, but in such
        # cases we simply count it as wrong.
        return 0.0

    correct_num = 0
    for gt in ground_truths:
        correct_num += calculate_normalized_exact_match_score(prediction, gt)

    min_annotator_agreements_needed = min(3, len(ground_truths))
    cur_acc = min(float(correct_num / min_annotator_agreements_needed), 1.0)
    return cur_acc


def calculate_exact_match_accuracy_from_records(records: list[dict[str, Any]]) -> float:
    # This was switched over in c967c2bfc3fb to use the VQAv2 style score, it should
    # not affect scores except for normalization, but be aware in case you see
    # any discrepancies.
    total = len(records)
    correct = 0
    for record in records:
        prediction = record["result"]
        ground_truth = record["label"]
        score = calculate_vqa_v2_exact_match_score(prediction, ground_truth)
        # if record["result"] == record["label"]:
        #     correct += 1
        if score > 0:
            correct += 1
    return correct / total


def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Returns the unbiased estimate of pass@$k$ as described in
    "Evaluating Large Language Models Trained on Code" in
    https://arxiv.org/abs/2107.03374.

    Parameters:
        n: int - total number of samples
        c: int - number of correct samples
        k: int - k in pass@$k$

    Returns:
        float - unbiased estimate of pass@$k$
    """

    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def calculate_unbiased_pass_at_k_from_evaluation_records_micro(
    records, k: int = 1
) -> dict[str, Union[float, dict[str, float]]]:
    df = pd.DataFrame(records)
    df["question_id"] = df["question_id"].astype(str)
    attempts_for_each_question = df.groupby("question_id")
    pass_at_k_proba_for_each_question: dict[str, float] = dict()
    for question_id, attempts in attempts_for_each_question:
        total_num_of_samples = len(attempts)
        num_of_correct_samples = (attempts["result"] == attempts["label"]).sum()
        pass_at_k = unbiased_pass_at_k(
            n=total_num_of_samples, c=num_of_correct_samples, k=k
        )
        pass_at_k_proba_for_each_question[str(question_id)] = pass_at_k
    micro_average = np.mean(list(pass_at_k_proba_for_each_question.values()))
    return {
        "pass_at_k": float(micro_average),
        "per_question": pass_at_k_proba_for_each_question,
    }


def naive_pass_at_k(
    solutions_for_problem: Sequence[bool], n: Optional[int] = None, k: int = 1
) -> bool:
    """

    Parameters:
        solutions_for_problem: Sequence[bool] - a sequence of booleans, one for each
            attempt at a solution, indicating whether that solution was correct.
        n: Optional[int] - the number of samples to draw. If None, defaults to the
            length of solutions_for_problem.
        k: int - k in pass@k

    Returns:
        bool - whether the problem was considered solved.
    """

    if n is None:
        n = len(solutions_for_problem)

    # We can't estimate pass@k if we have fewer than k samples.
    if k > n:
        raise ValueError(f"n must be at least k. Got n={n}, k={k}.")

    # Grab the first k samples.
    samples = solutions_for_problem[:k]

    # If any of the first k samples is a solution, the problem is solved.
    if any(samples):
        return True

    # Otherwise, the problem is not solved.
    return False


def calculate_naive_pass_at_k_from_evaluation_records(
    records, n: Optional[int] = None, k: int = 1
) -> dict[str, Union[float, dict[str, bool]]]:
    """
    Returns the "intuitive" definition of pass@k, which is:

    For each problem, we sample n code solutions and then select k of them for
    evaluation. If any of the k code solutions passes all ground truth test cases, the
    problem is considered solved. Then pass@k is the percentage of solved problems.

    (CodeT: Code Generation with Generated Tests)

    Parameters:
        records: list[dict[str, Any]] - a list of evaluation records.
        n: Optional[int] - the number of samples to draw. If None, defaults to the
            length of solutions_for_problem.
        k: int - k in pass@k
    Returns:
        float - the fraction of problems that were solved in k attempts.
    """
    df = pd.DataFrame(records)
    df["question_id"] = df["question_id"].astype(str)
    attempts_for_each_question = df.groupby("question_id")
    solution_did_pass_at_k: dict[str, bool] = dict()
    for question_id, attempts in attempts_for_each_question:
        # Precondition: check that the number of attempts is at least k.
        if len(attempts) < k:
            logger.warning(
                "Expected at least {} attempts for question_id={}"
                ", but got {}."
                " Skipping.",
                k,
                question_id,
                len(attempts),
            )
            continue

        # Convert this to a vanilla numpy array.
        solutions_for_problem = cast(
            Sequence[bool], (attempts["result"] == attempts["label"]).to_numpy()
        )
        solution_did_pass_at_k[str(question_id)] = naive_pass_at_k(
            solutions_for_problem, n=n, k=k
        )

    # Return the fraction of problems that were solved in k attempts.
    pass_at_k = sum(list(solution_did_pass_at_k.values())) / len(solution_did_pass_at_k)
    return {
        "pass_at_k": pass_at_k,
        "per_question": solution_did_pass_at_k,
    }


def preflight_checks_for_evaluation_records(
    records: list[dict[str, Any]], ks: Iterable[int]
) -> Iterable[int]:
    # Check how many attemps we have for each question.
    series = pd.Series([record["question_id"] for record in records])
    value_counts = series.value_counts()
    value_counts_counts = pd.Series(value_counts).value_counts()
    logger.info(f"Distribution of attempts per question: {value_counts_counts}")

    # Check that we have at least k attempts for each question.
    for k in sorted(ks):
        if any(value_counts < k):
            questions_with_too_few_attempts = value_counts[value_counts < k]
            logger.warning(
                "Expected at least {} attempts for each question, but got less for questions {}.",
                k,
                questions_with_too_few_attempts,
            )

            minimum_number_of_attempts = max(questions_with_too_few_attempts)
            # Filter ks to only include those that are less than or equal to the
            # minimum number of attempts.
            ks = [k for k in ks if k <= minimum_number_of_attempts]
            ks = sorted(ks)
            logger.warning(
                "Filtering ks to only include those that are <= min number of attempts: {}",
                ks,
            )
            break
    return ks


def calculate_naive_pass_at_k_by_question_type_from_evaluation_records(
    records: list[dict[str, Any]], n: Optional[int] = None, k: int = 1
) -> dict[str, float]:
    # We will do this by calculating pass@k for each question type (which will
    # be a bool telling us if that question passed or not)  and then averaging
    # over all questions of that type.

    solution_did_pass_at_k = cast(
        # Note, we have to tell the type checker that this will be a dict[str, bool].
        dict[str, bool],
        calculate_naive_pass_at_k_from_evaluation_records(records, k=k)["per_question"],
    )

    # Now we group by question type and average the pass@k for each question type.
    records_for_calc = [
        {
            "question_id": r["question_id"],
            "pass_at_k": solution_did_pass_at_k[str(r["question_id"])],
            "question_type": r["question_type"],
        }
        for r in records
    ]

    df = pd.DataFrame(records_for_calc)
    df["question_id"] = df["question_id"].astype(str)
    pass_at_k_by_question_type: dict[str, float] = dict()
    for question_type, group in df.groupby("question_type"):
        pass_at_k_by_question_type[str(question_type)] = float(
            group["pass_at_k"].mean()
        )

    return pass_at_k_by_question_type


def calculate_exact_match_accuracy_by_question_type_from_evaluation_records(
    records: list[dict[str, Any]]
) -> dict[str, float]:
    """
    Returns the exact match accuracy for each question type.

    Parameters:
        records: list[dict[str, Any]] - a list of evaluation records.

    Returns:
        dict[str, float] - a dictionary mapping question type to exact match accuracy.
    """

    # The naive pass@1 simply returns the fraction of question that were solved
    # in one attempt, which is the exact match accuracy.
    return calculate_naive_pass_at_k_by_question_type_from_evaluation_records(
        records, k=1
    )


def calculate_mean_and_std(accuracies: list[float]) -> tuple[float, float]:
    """
    Calculate the mean accuracy and standard deviation for a list of accuracy numbers.

    Parameters:
    accuracies (List[float]): List of accuracy numbers for each run of a model.

    Returns:
    Tuple[float, float]: Mean accuracy and standard deviation.
    """
    n = len(accuracies)
    mean_accuracy = sum(accuracies) / n
    variance = sum((x - mean_accuracy) ** 2 for x in accuracies) / n
    std_dev = math.sqrt(variance)

    return mean_accuracy, std_dev


def calculate_simulated_acc_stddev_from_evaluation_records(
    records: list[dict[str, Any]]
):
    # This will tell us how many runs we can simulate, because we know how many times
    # each question_id has been seen.
    question_id_counts = Counter([str(record["question_id"]) for record in records])
    # The lowest number of times a question_id has been seen is the number of times
    # we can simulate.
    num_of_runs = min(question_id_counts.values())
    logger.info("Calculating accuracy / stddev from {} simulated runs", num_of_runs)

    # Make a dictionary mapping question_id to a list of records for that question_id.
    # This will make it easy to grab a record for a question_id.
    records_by_question_id: dict[str, list[dict[str, Any]]] = dict()
    for record in records:
        question_id = str(record["question_id"])
        if question_id not in records_by_question_id:
            records_by_question_id[question_id] = []
        records_by_question_id[question_id].append(record)

    # For each run, we go round-robin through the list of question ids and pop a record
    # off the list of records for that question_id. This will give us the right number
    # of simulated_runs.
    simulated_runs: list[list[dict[str, Any]]] = []
    for _ in range(num_of_runs):
        run: list[dict[str, Any]] = []
        for question_id, records in records_by_question_id.items():
            # Select a record from the list of records for this question_id.
            record = records.pop()
            run.append(record)
        simulated_runs.append(run)

    # Now we can calculate the accuracy for each run.
    acc_per_simulated_run = []
    for run in simulated_runs:
        acc_per_simulated_run.append(calculate_exact_match_accuracy_from_records(run))

    # Calculate the mean and standard deviation of the accuracies.
    mean_accuracy, std_dev = calculate_mean_and_std(acc_per_simulated_run)
    return {"mean_accuracy": mean_accuracy, "stddev": std_dev}


def calculate_all_metrics(
    records: list[dict[str, Any]], ks: Iterable[int] = (1, 2, 3, 4, 5)
) -> dict:
    # We either return both str - > float (e.g. "exact_match_accuracy" -> 0.5) or a
    # str -> dict[int, float] (e.g. "unbiased_pass_at_k" -> {1: 0.5, 2: 0.6, ...}).

    ks = preflight_checks_for_evaluation_records(records, ks)
    metrics: dict[str, Any] = dict()

    metrics["exact_match_accuracy"] = calculate_exact_match_accuracy_from_records(
        records
    )
    metrics.update(
        **calculate_simulated_acc_stddev_from_evaluation_records(records),
    )
    percentage_of_each_failure_mode = calculate_percentage_of_all_failure_modes(records)
    for failure_mode, percentage in percentage_of_each_failure_mode.items():
        metrics[f"{failure_mode.value}"] = percentage

    # ks = [1, 2, 3, 4, 5]
    # ks = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]
    unbiased_pass_at_k: dict[int, float] = dict()
    for k in ks:
        # Convince the type checker that this will be a float.
        pass_at_k = cast(
            float,
            calculate_unbiased_pass_at_k_from_evaluation_records_micro(records, k=k)[
                "pass_at_k"
            ],
        )
        unbiased_pass_at_k[k] = pass_at_k
    naive_pass_at_k: dict[int, float] = dict()
    for k in ks:
        # Convince the type checker that this will be a float.
        pass_at_k = cast(
            float,
            calculate_naive_pass_at_k_from_evaluation_records(records, k=k)[
                "pass_at_k"
            ],
        )
        naive_pass_at_k[k] = pass_at_k
    metrics["unbiased_pass_at_k"] = unbiased_pass_at_k
    metrics["naive_pass_at_k"] = naive_pass_at_k

    if "question_type" in records[0]:
        pass_at_k_by_question_type: dict[int, dict[str, float]] = dict()
        # We can calculate metrics by question type. May not always be the
        # case, such as on datasets that do not have fine-grained question
        # types.
        for k in ks:
            pass_at_given_k_by_question_type = (
                calculate_naive_pass_at_k_by_question_type_from_evaluation_records(
                    records, k=k
                )
            )
            pass_at_k_by_question_type[k] = pass_at_given_k_by_question_type
        metrics["pass_at_k_by_question_type"] = pass_at_k_by_question_type
    return metrics


class VqaMetricsCalculator:
    def __init__(self, ks: Sequence[int]):
        self.ks = ks

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        return calculate_all_metrics(records, ks=self.ks)

    @classmethod
    def build_and_set_ks_from_it_per_record(cls, iterations_per_record: int):
        # The ks are the number of attempts we will use to calculate pass@k.
        # If we have 10 attempts per record, we can calculate pass@1, pass@2, ...
        # pass@10. so we can automatically set the ks if we are given the number
        # of attempts per record. This calculation is easy to do in Python, but
        # nontrivial to do with Hydra config interpolation, so we provide this
        # function to automatically set the ks.
        ks = list(range(1, iterations_per_record + 1))
        logger.info("Building {} with ks={}", cls.__name__, ks)
        return cls(ks=ks)


class VqaRunningMetricsFormatter:
    def __init__(self):
        self.num_correct = 0
        self.num_seen = 0

    def append(self, artifact: dict[str, Any]):
        prediction = artifact["result"]
        ground_truth = artifact["label"]
        vqa_ems = calculate_vqa_v2_exact_match_score(prediction, ground_truth)
        if vqa_ems:
            self.num_correct += 1
        self.num_seen += 1

    def log(self):
        with logging_redirect_tqdm():
            logger.info(
                "Running accuracy: {}",
                self.num_correct / self.num_seen,
            )


def bayesian_probability_better(
    accuracy1: float, accuracy2: float, sample_size: int, samples: int = 10000
) -> float:
    # Convert accuracies to Beta distribution parameters
    alpha1, beta1 = accuracy1 * sample_size, (1 - accuracy1) * sample_size
    alpha2, beta2 = accuracy2 * sample_size, (1 - accuracy2) * sample_size

    # Sample from posterior distributions
    posterior_samples_a = beta.rvs(alpha1 + 1, beta1 + 1, size=samples)
    posterior_samples_b = beta.rvs(alpha2 + 1, beta2 + 1, size=samples)

    # Calculate proportion where Model A is better than Model B
    proportion_better = np.mean(posterior_samples_a > posterior_samples_b)

    return float(proportion_better)


def convert_xywh_to_x1y1x2y2(
    box: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """Converts bounding box from (x, y, w, h) to (x1, y1, x2, y2) format."""
    x, y, w, h = box
    return (x, y, x + w, y + h)


def calculate_iou(box1, box2) -> float:
    """Calculates Intersection over Union (IoU) between two boxes."""
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2

    xi1 = max(x1, x1_gt)
    yi1 = max(y1, y1_gt)
    xi2 = min(x2, x2_gt)
    yi2 = min(y2, y2_gt)

    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0

    return iou


def calculate_f1_and_iou(
    predicted_boxes: list[tuple],
    ground_truth_boxes: list[tuple],
    iou_threshold: float = 0.5,
) -> tuple[float, list[float]]:
    """
    Calculates mean Average Precision (mAP) and mean IoU for multiple bounding boxes.

    The bounding boxes have to be in x_1, y_1, x_2, y_2 format, which is the same
    as the upper, left, right, lower format produced by the agentic model, but _different_
    from the format the ground-truth bounding boxes of Omnilabel are stored in.

    When evaluating on an object detection dataset such as Omnilabel, it is intended to
    be used on a single (description, image) pair.

    Parameters
    ------------
    predicted_boxes: The list of bounding boxes predicted by the model for the
        a (image, description) pair.
    ground_truth_boxes: The list of ground-truth bounding boxes for an
        (image, description) pair.
    """
    all_ious = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    num_annotations = len(ground_truth_boxes)
    ground_truth_box_matched = [False] * num_annotations

    for pred_box in predicted_boxes:
        max_iou = 0.0
        box_was_matched = False
        # Try and match this predicted box to a ground truth box.
        # If we cannot find a match, it is a false positive.
        for gt_box_idx, gt_box in enumerate(ground_truth_boxes):
            # If this ground truth box has already been matched
            # to a predicted box, skip it.
            if ground_truth_box_matched[gt_box_idx]:
                continue
            else:
                iou = calculate_iou(pred_box, gt_box)
                max_iou = max(max_iou, iou)
                # If the IoU is above the threshold, we have a true positive.
                if max_iou >= iou_threshold:
                    true_positives += 1
                    # Mark this ground truth box as matched, so we don't
                    # match it to another predicted box.
                    ground_truth_box_matched[gt_box_idx] = True
                    box_was_matched = True
                    break
        if not box_was_matched:
            false_positives += 1
        all_ious.append(max_iou)

    # false_negatives += num_annotations - true_positives
    false_negatives = len([_ for _ in ground_truth_box_matched if not _])

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives > 0
        else 0
    )

    ap = precision * recall / (precision + recall) * 2 if precision + recall > 0 else 0
    return ap, all_ious


def is_sequence_of_sequences_of_four_numeric(input_value) -> bool:
    """Check if input is a sequence of sequences, each containing four numeric items."""
    if not isinstance(input_value, Sequence):
        return False

    for sub_sequence in input_value:
        if not isinstance(sub_sequence, Sequence):
            return False
        if len(sub_sequence) != 4:
            return False
        if not all(isinstance(item, Number) for item in sub_sequence):
            return False

    return True


def calculate_macro_f1_and_mean_iou_from_evaluation_records(
    records: list[dict],
) -> dict[str, float]:
    f1_scores = []
    iou_scores = []

    for record in records:
        predicted_boxes = record["result"]
        ground_truth_boxes = tuple(record["label"])

        if isinstance(predicted_boxes, str):
            try:
                predicted_boxes=ast.literal_eval(predicted_boxes)
            except:
                continue

        if not isinstance(predicted_boxes, list):
            predicted_boxes = [predicted_boxes]
        if not isinstance(ground_truth_boxes, list):
            ground_truth_boxes = [ground_truth_boxes]

        f1: float
        ious: list[float]
        if not is_sequence_of_sequences_of_four_numeric(predicted_boxes):
            f1, ious = 0.0, []
        else:
            f1, ious = calculate_f1_and_iou(predicted_boxes, ground_truth_boxes)
        f1_scores.append(f1)
        iou_scores.extend(ious)
    macro_f1 = float(np.mean(f1_scores))
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0

    return {
        "macro_f1": macro_f1,
        "mean_iou": mean_iou,
    }


def calculate_percentage_malformed_object_detection_outputs(
    records: list[dict],
) -> float:
    total = len(records)
    count = 0
    for record in records:
        if not is_sequence_of_sequences_of_four_numeric(record["result"]):
            count += 1
    return count / total


def calculate_simulated_macro_f1_and_mean_iou_from_evaluation_records(
    records: list[dict],
):
    # This will tell us how many runs we can simulate, because we know how many times
    # each question_id has been seen.
    question_id_counts = Counter([str(record["question_id"]) for record in records])
    # The lowest number of times a question_id has been seen is the number of times
    # we can simulate.
    num_of_runs = min(question_id_counts.values())
    logger.info("Calculating accuracy / stddev from {} simulated runs", num_of_runs)

    # Make a dictionary mapping question_id to a list of records for that question_id.
    # This will make it easy to grab a record for a question_id.
    records_by_question_id: dict[str, list[dict[str, Any]]] = dict()
    for record in records:
        question_id = str(record["question_id"])
        if question_id not in records_by_question_id:
            records_by_question_id[question_id] = []
        records_by_question_id[question_id].append(record)

    # For each run, we go round-robin through the list of question ids and pop a record
    # off the list of records for that question_id. This will give us the right number
    # of simulated_runs.
    simulated_runs: list[list[dict[str, Any]]] = []
    for _ in range(num_of_runs):
        run: list[dict[str, Any]] = []
        for question_id, records in records_by_question_id.items():
            # Select a record from the list of records for this question_id.
            record = records.pop()
            run.append(record)
        simulated_runs.append(run)

    # Now we can calculate the accuracy for each run.
    metrics_per_simulated_run = []
    for run in simulated_runs:
        metrics_per_simulated_run.append(
            calculate_macro_f1_and_mean_iou_from_evaluation_records(run)
        )

    mean_aps = [run["macro_f1"] for run in metrics_per_simulated_run]
    mean_ious = [run["mean_iou"] for run in metrics_per_simulated_run]

    mean_ap, std_dev_ap = calculate_mean_and_std(mean_aps)
    mean_iou, std_dev_iou = calculate_mean_and_std(mean_ious)

    return {
        "macro_f1": mean_ap,
        "stddev_f1": std_dev_ap,
        "mean_iou": mean_iou,
        "stddev_iou": std_dev_iou,
    }


def calculate_all_metrics_object_detection(
    records: list[dict[str, Any]]
) -> dict[str, Any]:
    metrics: dict[str, Any] = dict()
    metrics.update(
        **calculate_simulated_macro_f1_and_mean_iou_from_evaluation_records(records),
    )

    percentage_of_each_failure_mode = calculate_percentage_of_all_failure_modes(records)
    for failure_mode, percentage in percentage_of_each_failure_mode.items():
        metrics[f"{failure_mode.value}"] = percentage
    metrics["percentage_malformed_outputs"] = (
        calculate_percentage_malformed_object_detection_outputs(records)
    )
    return metrics


class ObjectDetectionMetricsCalculator:
    def __init__(self):
        pass

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        return calculate_all_metrics_object_detection(records)


class RunningObjectDetectionMetricsFormatter:
    def __init__(self):
        # Maintain a running average of mAP and mIoU.
        self.average_precision_sum = 0
        self.iou_sum = 0
        self.records_seen = 0

    def append(self, record: dict[str, Any]):
        metrics = calculate_macro_f1_and_mean_iou_from_evaluation_records([record])
        average_precision_for_record = metrics["macro_f1"]
        average_iou_for_record = metrics["mean_iou"]

        self.average_precision_sum += average_precision_for_record
        self.iou_sum += average_iou_for_record
        self.records_seen += 1

    @property
    def running_mAP(self) -> float:
        return self.average_precision_sum / self.records_seen

    @property
    def running_mIoU(self) -> float:
        return self.iou_sum / self.records_seen

    def log(self):
        with logging_redirect_tqdm():
            logger.info(
                "Running macro-F1: {:.3f}, Running mIoU: {:.3f}",
                self.running_mAP,
                self.running_mIoU,
            )


def apply_decorator_to_methods_or_functions(
    decorator: Callable, callables: list[Union[Callable, Any]]
) -> None:
    for callable_item in callables:
        if inspect.ismethod(callable_item):
            # It's a method of a class
            cls = callable_item.__self__
            method_name = callable_item.__name__
            if hasattr(cls, method_name):
                original_method = getattr(cls, method_name)
                if not hasattr(original_method, "_applied_decorators"):
                    original_method._applied_decorators = set()
                if decorator in original_method._applied_decorators:
                    continue
                decorated_method = decorator(original_method)
                decorated_method._applied_decorators = (
                    original_method._applied_decorators
                )
                decorated_method._applied_decorators.add(decorator)
                setattr(cls, method_name, decorated_method)
        elif inspect.isfunction(callable_item):
            # It's a standalone function
            if not hasattr(callable_item, "_applied_decorators"):
                callable_item._applied_decorators = set()
            if decorator in callable_item._applied_decorators:
                continue
            decorated_function = decorator(callable_item)
            update_wrapper(decorated_function, callable_item)
            decorated_function._applied_decorators = callable_item._applied_decorators
            decorated_function._applied_decorators.add(decorator)
            globals()[callable_item.__name__] = decorated_function


MethodName = NewType("MethodName", str)
FunctionName = NewType("FunctionName", str)
CallableName = TypeVar("CallableName", MethodName, FunctionName)


class TracedFunctionCall(BaseModel):
    name: Union[MethodName, FunctionName]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    return_value: Any

    @classmethod
    def from_callable(
        cls,
        callable_type_constructor: Type[CallableName],
        callable_item: Callable,
        *args,
        **kwargs,
    ) -> "TracedFunctionCall":
        return_value = callable_item(*args, **kwargs)
        return cls(
            name=callable_type_constructor(callable_item.__qualname__),
            args=args,
            kwargs=kwargs,
            return_value=return_value,
        )


class SwitchableTrackingDecorator:
    _flag_planted_attr_name = "_switchable_tracking_decorator_applied"

    def __init__(
        self,
        func,
        tracking_sink: list[TracedFunctionCall],
        callable_type_constructor: Type[CallableName],
    ):
        # We first wrap the function with functools.wraps to preserve the
        # function's name and docstring. This has to come _before_ we set
        # the other attributes, because wraps(...) will overwrite them.
        wraps(func)(self)
        self.func = func
        self.active = True
        self.tracking_sink = tracking_sink
        self.callable_type_constructor: Type[CallableName] = callable_type_constructor  # type: ignore
        self.mark_func_as_decorated()

    def mark_func_as_decorated(self):
        setattr(self.func, self._flag_planted_attr_name, True)

    def __call__(self, *args, **kwargs):
        if self.active:
            # Execute the function and create a TracedFunctionCall.
            traced_function_call = TracedFunctionCall.from_callable(
                self.callable_type_constructor, self.func, *args, **kwargs
            )
            # Send the TracedFunctionCall to the tracking sink.
            self.tracking_sink.append(traced_function_call)
            result = traced_function_call.return_value
        else:
            result = self.func(*args, **kwargs)

        # Mark the function as decorated
        # self.func._is_decorated = True
        return result

    def deactivate(self):
        self.active = False
        del self.tracking_sink
        setattr(self.func, self._flag_planted_attr_name, False)


class ModuleTracer:
    _instance = None
    _reference_count = 0

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModuleTracer, cls).__new__(cls)
        # TODO: We increase the refcount whwnever we create a new instance,
        # but we only decrease it when we exit the context manager. This means
        # that if we create a new instance without ever calling __enter__ or
        # __exit__, we will never decrement the refcount. This class needs to be
        # engineered better. For example, we should perhaps _only_ allow adding
        # class methods to trace in the __enter__ method, and then we can guarantee
        # that we will always decrement the refcount in __exit__.
        cls._instance._increase_ref_count()
        return cls._instance

    def __init__(
        self,
    ) -> None:
        if hasattr(self, "_initialized"):
            return
        else:
            self._initialized = True
            self.called_functions: list[TracedFunctionCall] = []
            # We handle class methods and functions separately, because we want
            # to be able to deactivate tracking when we exit the context manager.
            # We can do this for class methods by replacing the class method with
            # the original class method. However, we cannot do this for functions,
            # because it is not guaranteed we can find the original function's scope
            # and replace it.
            self.class_methods_to_trace_stack: dict[int, list[Callable]] = defaultdict(
                lambda: []
            )
            self.original_class_methods: dict[MethodName, Callable] = dict()
            self.functions_to_trace_stack: dict[
                int, list[SwitchableTrackingDecorator]
            ] = defaultdict(lambda: [])

    def _filter_out_methods_that_are_already_tracked(
        self, class_methods: list[Callable]
    ):
        return [
            method
            for method in class_methods
            if MethodName(method.__qualname__) not in self.original_class_methods
        ]

    def add_class_methods_to_trace(self, class_methods: list[Callable]) -> None:
        self.owned_class_methods.extend(
            self._filter_out_methods_that_are_already_tracked(class_methods)
        )

    def trace(self, callable_: Callable) -> Callable:
        traced_function = SwitchableTrackingDecorator(
            callable_, self.called_functions, FunctionName
        )
        self.owned_functions.append(traced_function)
        return traced_function

    def __enter__(self) -> "ModuleTracer":
        owned_callables = self.owned_class_methods
        for callable_item in owned_callables:
            key = MethodName(callable_item.__qualname__)
            self.original_class_methods[key] = callable_item
            cls_reference = inspect._findclass(callable_item)  # type: ignore
            method_name = callable_item.__name__
            decorated_method = self._wrap_callable_with_tracker(
                callable_item, MethodName
            )
            setattr(cls_reference, method_name, decorated_method)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._teardown_tracking_for_class_methods()
        self._teardown_tracking_for_functions()
        self.__class__._reference_count -= 1
        self._reset_singleton_on_refcount_zero()

    def _reset_singleton_on_refcount_zero(self):
        # Remove the singleton instance. This should only happen
        # when we are exiting the context manager and there are no
        # active scopes above us that have instantiated a ModuleTracer.
        # Example:
        # with ModuleTracer(...) as module_tracer:
        #     ...
        #     with ModuleTracer(...) as module_tracer2:
        #         ...  # We SHOULD NOT remove the singleton instance here.
        if self.__class__._reference_count <= 0:
            self.__class__._instance = None

    def _switch_off_tracking_deco_from_owned_functions(self):
        for callable_item in self.owned_functions:
            callable_item.deactivate()

    def _wrap_callable_with_tracker(
        self, callable_: Callable, callable_type_constructor: Type[CallableName]
    ) -> Callable:
        @wraps(callable_)
        def wrapper(*args, **kwargs):
            called_functions = self.called_functions
            traced_function_call = TracedFunctionCall.from_callable(
                callable_type_constructor, callable_, *args, **kwargs
            )
            called_functions.append(traced_function_call)
            return traced_function_call.return_value

        return wrapper

    def _increase_ref_count(self):
        self.__class__._reference_count += 1

    @property
    def owned_class_methods(self) -> list[Callable]:
        reference_count = self.__class__._reference_count
        return self.class_methods_to_trace_stack[reference_count]

    @property
    def owned_functions(self) -> list[SwitchableTrackingDecorator]:
        reference_count = self.__class__._reference_count
        return self.functions_to_trace_stack[reference_count]

    def _teardown_tracking_for_class_methods(self):
        for callable_item in self.owned_class_methods:
            key = MethodName(callable_item.__qualname__)
            original_method = self.original_class_methods[key]
            cls_reference = inspect._findclass(callable_item)  # type: ignore
            method_name = callable_item.__name__
            setattr(cls_reference, method_name, original_method)
        reference_count = self.__class__._reference_count
        self.class_methods_to_trace_stack.pop(reference_count)

    def _teardown_tracking_for_functions(self):
        for callable_item in self.owned_functions:
            callable_item.deactivate()
        reference_count = self.__class__._reference_count
        self.functions_to_trace_stack.pop(reference_count)

        
def calculate_exact_match_group_accuracy_from_records(records: list[dict[str, Any]]) -> float:
    total = len(records)
    correct = 0
    grounp_acc_counter = {}
    image_acc_counter = {}
    query_acc_counter = {}
    for record in records:
        if record["group"] not in grounp_acc_counter:
            grounp_acc_counter[record["group"]] = {"correct": 0, "total": 0}
        if record["image_id"] not in image_acc_counter:
            image_acc_counter[record["image_id"]] = {"correct": 0, "total": 0}
        if record["query"] not in query_acc_counter:
            query_acc_counter[record["query"]] = {"correct": 0, "total": 0}
 
        grounp_acc_counter[record["group"]]["total"] += 1
        image_acc_counter[record["image_id"]]["total"] += 1
        query_acc_counter[record["query"]]["total"] += 1
 
        prediction = record["result"]
        ground_truth = record["label"]
        score = calculate_vqa_v2_exact_match_score(prediction, ground_truth)
        if score > 0:
            correct += 1
            grounp_acc_counter[record["group"]]["correct"] += 1
            image_acc_counter[record["image_id"]]["correct"] += 1
            query_acc_counter[record["query"]]["correct"] += 1
 
    acc = correct / total
    i_acc = sum([elem["correct"] == elem["total"] for elem in image_acc_counter.values()]) / len(image_acc_counter)
    q_acc = sum([elem["correct"] == elem["total"] for elem in query_acc_counter.values()]) / len(query_acc_counter)
    g_acc = sum([elem["correct"] == elem["total"] for elem in grounp_acc_counter.values()]) / len(grounp_acc_counter)
    return acc, i_acc, q_acc, g_acc