from src.instrumentation import calculate_macro_f1_and_mean_iou_from_evaluation_records
import numpy as np
from loguru import logger
from functools import wraps


# Define a decorator that prints the number of records that were filtered out.
# Make sure to use functools.wraps.
def log_number_of_records_filtered_out(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        num_records_before = len(args[1])
        records_after = func(*args, **kwargs)
        num_records_after = len(records_after)
        logger.info(
            "Filtered out {} records out of {} ({:.2f}%)",
            num_records_before - num_records_after,
            num_records_before,
            100 * (num_records_before - num_records_after) / num_records_before,
        )
        return records_after

    return wrapper


class FilterByExactMatch:
    @log_number_of_records_filtered_out
    def __call__(self, records: list[dict]) -> list[dict]:
        return [record for record in records if record["result"] == record["label"]]


class FilterByF1Threshold:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def get_f1_for_record(self, record: dict) -> float:
        # Macro-F1 for a single record is just the F1 for that record.
        metrics = calculate_macro_f1_and_mean_iou_from_evaluation_records([record])
        return metrics["macro_f1"]

    @log_number_of_records_filtered_out
    def __call__(self, records: list[dict]) -> list[dict]:
        return [
            record
            for record in records
            if self.get_f1_for_record(record) >= self.threshold
        ]


class FilterByF1Percentile:
    def __init__(self, percentile: float):
        self.percentile = percentile

    def get_f1_for_record(self, record: dict) -> float:
        # Macro-F1 for a single record is just the F1 for that record.
        metrics = calculate_macro_f1_and_mean_iou_from_evaluation_records([record])
        return metrics["macro_f1"]

    @log_number_of_records_filtered_out
    def __call__(self, records: list[dict]) -> list[dict]:
        f1s = np.array([self.get_f1_for_record(record) for record in records])
        threshold = np.percentile(f1s, self.percentile)
        return [
            record for record in records if self.get_f1_for_record(record) >= threshold
        ]
