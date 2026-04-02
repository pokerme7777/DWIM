"""
Read the JSON file of the output produced by the GQA evaluation scripts. Compute more detailed
metrics that the original scripts do not compute.
"""

import json
import argparse
import pandas as pd



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)

    args = parser.parse_args()

    with open(args.input_path, "r") as f:
        artifacts = json.load(f)

    df = pd.DataFrame(artifacts.values())

    # Calculate the accuracy _only_ for questions that we generated and
    # executed a valid program for.
    filtered  = df[df['result'].notnull()]
    accuracy = (filtered['result'] == filtered['label']).mean()

    print(f"Accuracy for questions with valid programs: {accuracy}")
