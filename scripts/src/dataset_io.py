import pandas as pd
from datasets import load_dataset
from .pale_giant_utils import JsonlIoHandler
from torch.utils.data import Dataset
import os
from typing import Any, Sequence, cast, Optional, Union, Callable
import json
from omegaconf import DictConfig, ListConfig, OmegaConf
from collections.abc import Iterable
from beartype.door import is_bearable
from loguru import logger


def load_curated_gqa_val_subset(
    selected_questions_csv_path: str = "fill_the_path_to_gqa_val_subset_csv"
) -> list[dict[str, Any]]:
    # Load the qids of questions from GQA we will evaluate.
    selected_gqa_questions = pd.read_csv(selected_questions_csv_path)
    # Coerce question_id to str for consistency.
    selected_gqa_questions["question_id"] = selected_gqa_questions[
        "question_id"
    ].astype(str)
    selected_qids = set(selected_gqa_questions.question_id)
    gqa_val = load_dataset("Graphcore/gqa", split="validation")
    gqa_val_subset = gqa_val.filter(lambda x: str(x["question_id"]) in selected_qids)

    # The method originally terminated here. When we first wrote this
    # we were using the GQA version hosted on Huggingface. But it was
    # incomplete and missing some metadata, so we moved away from it.
    # Later on, we realized we needed the question_type field for
    # metrics, so we are adding it in here.

    map_question_id_to_question_type: dict[str, str] = dict()
    for record in selected_gqa_questions.to_dict(orient="records"):
        map_question_id_to_question_type[str(record["question_id"])] = record[
            "types.detailed"
        ]
    gqa_val_subset = gqa_val_subset.to_pandas().to_dict(orient="records")  # type: ignore
    gqa_val_subset = cast(list[dict[str, Any]], gqa_val_subset)
    for record in gqa_val_subset:
        record["question_type"] = map_question_id_to_question_type[
            str(record["question_id"])
        ]

    return gqa_val_subset


class JsonlDatasetWithImageRoot(Dataset):
    def __init__(self, image_root, jsonl_path, postfix=None):
        self.image_root = image_root
        self.jsonl_path = jsonl_path
        self.records = JsonlIoHandler(self.jsonl_path).read_all()
        # self.postfix=postfix

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        record["image_id"] = os.path.join(self.image_root, record["image_id"])
        #  # add postfix to image_id
        # if self.postfix:
        #     record["image_id"] = record["image_id"]+self.postfix
        return record


##new loader for post processing captioning##
class JsonlDatasetWithImageRoot_with_postcaption(Dataset):
    def __init__(self, image_root, jsonl_path):
        self.image_root = image_root
        self.jsonl_path = jsonl_path
        self.records = JsonlIoHandler(self.jsonl_path).read_all()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        record["image_id"] = os.path.join(self.image_root, record["image_id"])

        description_out = record['grit']['descriptions']
        tag2text = record['tag2text']
        description_out.append(tag2text['caption'])

        # record['prost_captioning'] = {
        #     'descriptions': description_out,
        #     'tag': tag2text['tags']
        # }
        
        description_out = '\n'.join(description_out)
        tag2text['tags'] = tag2text['tags'].replace(' | ', '\n')
        record['prost_captioning'] = f"<Image description>\n{description_out}\n</Image description>\n\n<Tag name on image>\n{tag2text['tags']}\n</Tag name on image>"

        return record

class VqaDatasetWithImageRoot(Dataset):
    def __init__(self, image_root, json_records_path, slice: Optional[slice] = None):
        self.image_root = image_root
        self.json_records_path = json_records_path
        if "jsonl" in self.json_records_path:
            # Load JSON Lines file
            self.records = []
            with open(self.json_records_path, "r") as f:
                for line in f:
                    self.records.append(json.loads(line))
        else:
            with open(self.json_records_path, "r") as f:
                self.records = json.load(f)
        if slice is not None:
            self.records = self.records[slice]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        for_return = {
            "image_id": os.path.join(self.image_root, record["image"]),
            "question": record["question"],
            "label": record["answer"],
            "question_type": record["dataset"],
            "question_id": record["question_id"],
        }
        return for_return


# TODO: This is horrible. We should change this... to something.
# Maybe use a dataclass.
def parse_train_records_path_compatibility_shim(
    train_records_path: Union[str, list[str], list[dict]]
) -> Union[list[DictConfig], ListConfig]:
    """
    Acceptable options are:
    - A string ending with .jsonl
    - A string ending with .yaml
    - A list of strings ending with .jsonl
    - A list of config dicts, each already having a path and sieve specified
    """
    default_sieve = OmegaConf.create({"_target_": "src.filtering.FilterByExactMatch"})
    if isinstance(train_records_path, str):
        # E.g. "foo.jsonl", which was the very first format we used.
        if train_records_path.endswith(".jsonl"):
            logger.info(
                "A single JSONL file {} was specified"
                " for train_records_path, so we will use"
                " the default sieve (FilterByExactMatch).",
                train_records_path,
            )
            return [
                OmegaConf.create({"path": train_records_path, "sieve": default_sieve})
            ]
        # E.g. "foo.yaml", which is the format we use now, in which
        # we can specify multiple records in a YAML file, so the file has
        # something which looks like:
        # - path: foo.jsonl
        #   sieve:
        #     _target_: src.FilterByExactMatch or whatever
        # - path: bar.jsonl
        #   sieve:
        #     _target_: src.FilterByExactMatch or whatever
        elif train_records_path.endswith(".yaml"):
            logger.info(
                "A single YAML file {} was specified"
                " for train_records_path, so we will load train records from it.",
                train_records_path,
            )
            with open(train_records_path, "r") as f:
                train_records = OmegaConf.load(f)
            if isinstance(train_records, ListConfig):
                logger.info(
                    "Loaded {} train record paths from {}",
                    len(train_records),
                    train_records_path,
                )
                return train_records
            else:
                raise ValueError(
                    f"""Was given {train_records_path} but it is a """
                    f"""{type(train_records)} instead of a list."""
                )
        # Whatever we got passed in was wrong.
        else:
            raise ValueError(
                f"""Was given {train_records_path} but it does not end """
                f"""with .jsonl or .yaml."""
            )
    # This is already a dictconfig, such as in the case we directly specify
    # something like:
    # train_records_path:
    #   path: foo.jsonl
    #   sieve:
    #     _target_: src.FilterByExactMatch or whatever
    elif isinstance(train_records_path, DictConfig):
        return OmegaConf.create([train_records_path])
    elif isinstance(train_records_path, Sequence):
        # This is only valid if it is a list of strings ending with .jsonl, which was
        # the second format we used. So something like:
        # train_records_path:
        #   - foo.jsonl
        #   - bar.jsonl
        logger.info(
            "A list of train record paths was specified"
            " for train_records_path, so we will use"
            " the default sieve (FilterByExactMatch).",
        )
        if all(isinstance(_, str) and _.endswith(".jsonl") for _ in train_records_path):
            return OmegaConf.create(
                [{"path": _, "sieve": default_sieve} for _ in train_records_path]
            )
        # Assume that we have a list of dicts, each of which already has a path and sieve
        # specified. So we do nothing here.
        elif all(isinstance(_, DictConfig) for _ in train_records_path):
            return OmegaConf.create(train_records_path)
        else:
            raise ValueError(
                f"""Was given {train_records_path} but it is a """
                f"""{type(train_records_path)} which I have no branch for."""
            )
    else:
        raise ValueError(
            f"""Was given {train_records_path} but it is a """
            f"""{type(train_records_path)} instead of a str or list[str]."""
        )
