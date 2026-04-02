import argparse
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="The path to the adapter")
    args = parser.parse_args()

    with open(os.path.join(args.adapter_path, "adapter_config.json"), "r") as f:
        config = json.load(f)

    base_model = AutoModelForCausalLM.from_pretrained(
        config["base_model_name_or_path"],
        torch_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)

    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model = model.merge_and_unload()

    output_path = args.adapter_path.rstrip("/") + "_merged"
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
