# GQA Workflow Release Guide

This public release focuses on the workflow around these scripts:

- `bash_script/001_collect_llama31_SI_NF_ALLTOOL_GQA_train10.sh`
- `bash_script/001_for_preprocessing.sh`
- `bash_script/001_training_lora64_GQA.sh`
- `bash_script/001_merge_lora64_llama318b.sh`
- `bash_script/001_evaluate_llama31_SI_NF_ALLTOOL_GQA_test.sh`

All private information (paths, IPs, usernames) has been replaced with `fill_the_*` placeholders.

## 1. Environment Setup (Original-Style Cluster Flow)

This section keeps the same setup logic as the original English README, but sanitized for public release.

### 1.1 Log in to your cluster (if applicable)

```bash
ssh <your-username>@fill_the_slurm_submit_host
```

### 1.2 Create a workspace on shared storage

```bash
mkdir -p fill_the_shared_storage_root/$(whoami)
cd fill_the_shared_storage_root/$(whoami)
```

### 1.3 Install Miniconda on shared storage

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p fill_the_shared_storage_root/$(whoami)/miniconda3
conda init bash
```

Open a new shell after `conda init bash`.

### 1.4 Request an interactive GPU node for installation (optional)

```bash
srun --nodes=1 --ntasks=1 --export=ALL --cpus-per-gpu=4 --mem-per-gpu=32G --gres=gpu:1 --time=1:00:00 --constraint=fill_the_gpu_constraint --pty /bin/bash
```

### 1.5 Clone the repo and run project setup

```bash
git clone <your_github_repo_url>
cd <repo_name>
./setup.sh
```

`setup.sh` will:

- create a conda environment (default: `agent_study`)
- install dependencies from `requirements.txt`
- install this repository as a package (`pip install -e .`)

### 1.6 Configure cache-related environment variables

Set at least:

```bash
export AGENT_STUDY_CACHE="fill_the_path_to_cache_parent"
export OPENAI_API_KEY="fill_the_openai_api_key"
```

Recommended cache paths:

```bash
export HF_HOME="fill_the_path_to_hf_home"
export HF_DATASETS_CACHE="fill_the_path_to_hf_datasets_cache"
export TRANSFORMERS_CACHE="fill_the_path_to_hf_models_cache"
export TORCH_HOME="fill_the_path_to_torch_cache"
```

If SAM is used:

```bash
export SAM_CHECKPOINT_PATH="fill_the_path_to_sam_checkpoint"
```

Optional conda env overrides used by scripts:

```bash
export RUNTIME_CONDA_ENV="agent_study"
export TRAIN_CONDA_ENV="agent_study"
export MERGE_CONDA_ENV="agent_study"
```

### 1.7 Extra dependencies (install only if needed)

If missing-import errors appear, install:

```bash
pip install tenacity
pip install trl==0.9.6
pip install qwen-vl-utils==0.0.5
pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830
pip install -e git+https://github.com/IDEA-Research/GroundingDINO.git@main#egg=GroundingDINO
pip install --upgrade segments-ai scikit-image
```

## 2. Runtime Components

The core runtime stack is the same as the original project design:

- **Ray cluster** for distributed workers/actors.
- **Jupyter Kernel Gateway** for isolated code execution by the environment.
- **AgenticTaskRunner** (`neurips_prototyping/new_worker.py`) to orchestrate agent-environment steps.

## 3. Start Ray and Jupyter Infrastructure

## 3.1 Bring up a Ray head node

Example Slurm request (template):

```bash
srun --nodes=1 --ntasks=1 --export=ALL --cpus-per-gpu=4 --mem-per-gpu=32G --gres=gpu:4 --time=1-0 --constraint=fill_the_gpu_constraint --pty /bin/bash
```

Start Ray head:

```bash
ray start --head --port fill_the_ray_head_port
```

## 3.2 Bring up Ray worker nodes (optional, for multi-node)

On each worker node:

```bash
ray start --address=fill_the_ray_head_ip:fill_the_ray_head_port --num-gpus fill_the_num_gpus_on_this_node
```

## 3.3 Bring up a Jupyter Kernel Gateway node

Example CPU node request (template):

```bash
srun --nodes=1 --ntasks=1 --export=ALL --cpus-per-task=4 --mem-per-cpu=8G --time=1-0 --partition=cpu --pty /bin/bash
```

Attach this node to Ray if needed:

```bash
ray start --address=fill_the_ray_head_ip:fill_the_ray_head_port
```

Start Jupyter Kernel Gateway:

```bash
jupyter kernelgateway \
  --ip=0.0.0.0 \
  --port=fill_the_jupyter_gateway_port \
  --JupyterWebsocketPersonality.list_kernels=True
```

Set the same values in `configs/eval_prototype.yaml`:

- `jupyter_kernel_gateway.ip_address`
- `jupyter_kernel_gateway.port`

## 4. Required Placeholder Replacement

Replace `fill_the_*` values in these files before running:

- `bash_script/001_for_preprocessing.sh`
- `bash_script/001_training_lora64_GQA.sh`
- `bash_script/001_merge_lora64_llama318b.sh`
- `bash_script/001_collect_llama31_SI_NF_ALLTOOL_GQA_train10.sh`
- `bash_script/001_evaluate_llama31_SI_NF_ALLTOOL_GQA_test.sh`
- `configs/eval_prototype.yaml`
- `configs/dataset/gqa_train_balanced_subset_10.yaml`
- `configs/dataset/gqa_val_subset.yaml`

Most important placeholders:

- data paths: `fill_the_path_to_raw_dataset_jsonl`, `fill_the_path_to_training_jsonl`
- model paths: `fill_the_path_to_adapter_checkpoint`, `fill_the_path_to_merged_model_or_hf_model_id`
- Ray/Jupyter settings: `fill_the_ray_head_ip`, `fill_the_ray_head_port`, `fill_the_jupyter_gateway_ip`, `fill_the_jupyter_gateway_port`
- dataset config fields: `image_root`, `jsonl_path`

## 5. Run the Workflow

Recommended execution order:

```bash
bash bash_script/001_collect_llama31_SI_NF_ALLTOOL_GQA_train10.sh
bash bash_script/001_for_preprocessing.sh
bash bash_script/001_training_lora64_GQA.sh
bash bash_script/001_merge_lora64_llama318b.sh
bash bash_script/001_evaluate_llama31_SI_NF_ALLTOOL_GQA_test.sh
```

What each step does:

- `collect`: generates trajectory records.
- `for_preprocessing`: converts `records.jsonl` into training trajectory samples.
- `training`: runs SFT + LoRA training.
- `merge`: merges LoRA adapter into base model weights.
- `evaluate`: evaluates on validation set (`EVAL_MODEL_PATH` points to merged model path or model ID).

## 6. Script-Specific Notes

- `collect` and `evaluate` scripts keep `ray start --address=...` commented by default. Uncomment if your runtime node is not already attached to Ray.
- `001_for_preprocessing.sh` supports:
  - `--okvqa_image_root`
  - `--refcoco_image_root`
- `configs/dataset/*.yaml` uses `JsonlDatasetWithImageRoot`. Your JSONL should provide fields consumed by this workflow (for example `image_id`, `question`, `label`).

## 7. FAQ

### 7.1 Missing `openai` / `tenacity`

```bash
pip install openai tenacity
```

### 7.2 `conda: command not found`

Run `conda init bash`, then open a new shell.

### 7.3 Jupyter connection errors

Check:

- gateway process is running
- `configs/eval_prototype.yaml` has correct gateway IP/port
- worker nodes can reach the gateway host

## 8. Public Release Checklist

1. Ensure there are no real private paths, IPs, or tokens.
2. Keep `fill_the_*` placeholders in shared examples.
3. Do not commit local datasets, checkpoints, cache directories, or secrets.
