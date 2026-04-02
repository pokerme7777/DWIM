#!/bin/bash

source lib.sh

make_project_cache()(
  PROJECT_CACHE=$(get_project_cache)
  if [ -d "$PROJECT_CACHE" ]; then
    echo "Project cache already exists."
  else
    echo "Creating project cache."
    mkdir -p $PROJECT_CACHE
  fi

  # # Now symlink the cache folder to the cache under the project root.
  # PROJECT_CACHE_SYMLINK=$(python3 -m src.project_constants PROJECT_CACHE_LOCAL)
  # if [ -L "$PROJECT_CACHE_SYMLINK" ]; then
  #   echo "Project cache symlink already exists at $PROJECT_CACHE_SYMLINK."
  # else
  #   echo "Creating project cache symlink."
  #   ln -s $PROJECT_CACHE $PROJECT_CACHE_SYMLINK
  # fi

  # # Print the symlink using ls -l.
  # ls -l $PROJECT_CACHE_SYMLINK
)

create_conda_env()(
  # We're going to write an idempotent version of this below.
  # First we check if the conda env already exists.
  if conda env list | grep -q $(get_conda_env_name); then
    echo "Conda environment already exists."
    echo "If you want to remove it, run 'conda env remove -n $(get_conda_env_name)'"
  else
    echo "Creating conda environment."
    conda create -n $(get_conda_env_name) python=3.10 --yes
  fi
)


create_conda_env

echo "Activating the environment..."
eval "$(conda shell.bash hook)"
set_conda_environment

echo "Installing the requirements..."
pip install uv

uv pip sync requirements.txt
uv pip install -e .

make_project_cache