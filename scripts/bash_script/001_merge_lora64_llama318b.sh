#!/bin/bash
set -euo pipefail

source ~/.bashrc
RUNTIME_CONDA_ENV="${RUNTIME_CONDA_ENV:-agent_study}"
conda activate "$RUNTIME_CONDA_ENV"
python train_script_folder/merge_weights_bf16.py --adapter_path 'fill_the_path_to_adapter_checkpoint'
