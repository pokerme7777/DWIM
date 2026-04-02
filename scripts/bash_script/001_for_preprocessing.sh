#!/bin/bash
set -euo pipefail

source ~/.bashrc
RUNTIME_CONDA_ENV="${RUNTIME_CONDA_ENV:-agent_study}"
conda activate "$RUNTIME_CONDA_ENV"

python ./f_script/prototype_generate_trajectory_generator_skip_masking_err.py \
    --template_file_path '2025_01_25_all_tool_GQA_0shot.xml' \
    --ice_file_path '2024-10-04-super-Instruct_fortest_no_ice.yaml' \
    --raw_dataset_path 'fill_the_path_to_raw_dataset_jsonl' \
    --okvqa_image_root 'fill_the_path_to_okvqa_images' \
    --refcoco_image_root 'fill_the_path_to_refcoco_images' \
    --output_file_folder_name '001-pre_processing_trajectory' \
    --inference_records_folder_name '001_collect_llama31_SI_NF_ALLTOOL_GQA_train10' \
    --cache_parent_folder_name 'fill_the_path_to_cache_parent' \
    --instruction_mode
