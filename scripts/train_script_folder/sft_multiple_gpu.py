import argparse
from datasets import disable_caching, Dataset
from peft import LoraConfig, get_peft_model
import json
import torch
import os
import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM


# Disable caching for HF datasets
disable_caching()


if __name__ == "__main__":
    # ----- Parse Arguments -----
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True,
                        help="The training file")
    parser.add_argument("--output_dir", type=str, default="training_output",
                        help="The output directory to save the model")
    parser.add_argument("--model_name_", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="name of model and model dir")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=2,
                        help="num_train_epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="per_device_train_batch_size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="accumulation")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="lora rank")
    parser.add_argument("--lora_rank", type=int, default=16,
                        help="lora rank")
    parser.add_argument("--enable_lora", action="store_true", help="whether to use lora")
    parser.add_argument("--target_modules_full_flag", action="store_true", help="flag to full list of lora target")
    parser.add_argument("--output_model_folder_subname", type=str, required=True,
                        help="name of each different folder")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="lora dropout rate")
    
    args = parser.parse_args()
    # ----- Parse Arguments -----

    model_name = args.model_name_
    
    # ----- Load the model and tokenizer -----
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )
    if args.enable_lora:
        if args.target_modules_full_flag:
            target_modules_list = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        else:
            target_modules_list = ["q_proj", "v_proj"]
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=16,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules_list,
            use_rslora=True,
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        # ----- Load the model and tokenizer -----


    # ----- Prepare the dataset -----
    with open(args.train_file, "r") as f:
        data = [json.loads(line) for line in f]
    processed_data = {"text": []}
    for d in data:
        processed_data["text"].append(tokenizer.apply_chat_template(
            conversation=d,
            tokenize=False,
            add_generation_prompt=False,
        ))
    dataset = Dataset.from_dict(processed_data)
    # ----- Prepare the dataset -----


    # ----- Prepare the data collator -----
    # NOTE: This is specifically for Llama-3.1
    inst_tokens = tokenizer.encode("<|start_header_id|>user<|end_header_id|>", add_special_tokens=False)
    response_tokens = tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False)
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        instruction_template=inst_tokens,
        response_template=response_tokens,
    )
    # ----- Prepare the data collator -----

    # ----- SFT Arguments & Trainer -----
    log_name = f"{os.path.splitext(os.path.basename(os.path.dirname(args.train_file)))[0]}_{args.output_model_folder_subname}"
    output_dir = os.path.join(args.output_dir, log_name)
    os.makedirs(output_dir, exist_ok=True)

    training_args = SFTConfig(
        learning_rate=args.learning_rate,
        max_grad_norm=1.0,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        seed=3407,
        num_train_epochs=args.num_train_epochs,
        output_dir=output_dir,
        optim="adamw_torch",
        logging_steps=1,
        warmup_ratio=args.warmup_ratio,
        save_strategy="epoch",
        max_seq_length=8192,
        lr_scheduler_type="cosine",
        dataset_text_field="text",
        report_to="wandb"
    )
    
    if "Llama-3.1" in model_name:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )
    else:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=dataset,
        )
    # ----- SFT Arguments & Trainer -----


    # ----- Training -----
    if list(Path(training_args.output_dir).glob("chechpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_model(training_args.output_dir)
    # ----- Training -----
