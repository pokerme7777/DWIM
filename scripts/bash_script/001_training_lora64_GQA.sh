#!/bin/bash
set -euo pipefail

source ~/.bashrc
TRAIN_CONDA_ENV="${TRAIN_CONDA_ENV:-agent_study}"
MERGE_CONDA_ENV="${MERGE_CONDA_ENV:-agent_study}"
conda activate "$TRAIN_CONDA_ENV"

export WANDB_PROJECT='fill_the_wandb_project_name'

accelerate launch --config_file ./train_script_folder/deepspeed_zero3.yaml ./train_script_folder/sft_multiple_gpu.py \
    --train_file 'fill_the_path_to_training_jsonl' \
    --output_dir 'fill_the_path_to_output_dir' \
    --learning_rate 3e-5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --warmup_ratio 0.05 \
    --output_model_folder_subname 'fill_the_model_subname' \
    --num_train_epochs 6 \
    --lora_rank 64 \
    --enable_lora \
    --target_modules_full_flag

conda activate "$MERGE_CONDA_ENV"
python train_script_folder/merge_weights_bf16.py --adapter_path 'fill_the_path_to_adapter_checkpoint'
