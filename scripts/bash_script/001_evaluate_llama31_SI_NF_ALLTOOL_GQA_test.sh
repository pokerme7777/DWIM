#!/bin/bash
set -euo pipefail

source lib.sh
set_conda_environment
exit_if_only_summary_needed "$@"
print_output_folder
make_output_folder

RAY_HEAD_ADDRESS='fill_the_ray_head_ip:fill_the_ray_head_port'

OUTPUT_FOLDER=$(get_output_folder)
AGENT=llama31_8b_vllm_react_agent
EVAL_MODEL_PATH='fill_the_path_to_merged_model_or_hf_model_id'
DATASET=gqa_val_subset
PROMPTER=20250125-SI-NF-ALLTOOL-GQA-0shot-evaluate

TOOLS_MODULE_NAME=vqa_llava15

TENSOR_PARALLEL_SIZE=4
NUM_GPUS=6
RAY_NUM_WORKERS=5
echo "Using $RAY_NUM_WORKERS workers across $NUM_GPUS gpus."

# Exit if ctrl-c.
trap "exit" INT

# ray start --address=$RAY_HEAD_ADDRESS --num-gpus=$NUM_GPUS

python3 prototype.py output_folder=$OUTPUT_FOLDER \
    ray.num_workers=$RAY_NUM_WORKERS \
    agentic_task_runner/agent=$AGENT \
    agentic_task_runner/agent.model=$EVAL_MODEL_PATH \
    restart=true \
    experiment_namespace=$(get_script_name) \
    dataset=$DATASET \
    agentic_task_runner/agent/prompter=$PROMPTER \
    iterations_per_record=3 \
    agentic_task_runner/environment/module_specs=$TOOLS_MODULE_NAME \
    caption_first=1 \
    instructed_mode=1 \
    use_refered_answer=1 \
    NF_flag=1
