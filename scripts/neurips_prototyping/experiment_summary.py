from collections import Counter

import click
import numpy as np
from rich import print

from src.pale_giant_utils import JsonlIoHandler


@click.command()
@click.argument("experiment_path", type=str)
def main(experiment_path: str) -> None:
    io_handler = JsonlIoHandler(experiment_path)
    records = io_handler.read_all()

    # Count number of records
    num_records = len(records)
    print(f"Number of records: {num_records}")

    # Count the different failure modes.
    failure_modes: Counter = Counter()
    for record in records:
        failure_modes[record["failure_mode"]] += 1
    print(f"Failure modes: {failure_modes}")

    # Print summary statistics about the number of steps per episode
    # using numpy to compute them
    steps_per_episode = np.array([len(record["trajectory"]) for record in records])
    percentiles = np.percentile(steps_per_episode, [25, 50, 75, 90, 95, 99])
    for percentile, value in zip([25, 50, 75, 90, 95, 99], percentiles):
        print(f"Percentile {percentile}: num_steps={value}")


if __name__ == "__main__":
    main()
