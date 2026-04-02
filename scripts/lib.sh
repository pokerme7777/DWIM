get_project_name()(
  echo "study_on_agent"
)

get_conda_env_name()(
  echo "agent_study"
)

PROJECT_NAME=$(get_project_name)
CONDA_ENV_NAME=$(get_conda_env_name)

get_project_cache()(
  if [ -z "${AGENT_STUDY_CACHE:-}" ]; then
    echo "ERROR: AGENT_STUDY_CACHE is not set. Export AGENT_STUDY_CACHE=fill_the_path_to_cache_parent" >&2
    return 1
  fi
  echo "$AGENT_STUDY_CACHE"
)

set_conda_environment(){
  local ENVIRONMENT_NAME="$CONDA_ENV_NAME"
  if [ "${CONDA_DEFAULT_ENV:-}" = "$ENVIRONMENT_NAME" ]; then
    echo "conda environment already activated"
  else
    echo "activating conda environment"
    if ! command -v conda >/dev/null 2>&1; then
      echo "ERROR: conda command not found in PATH." >&2
      return 1
    fi
    eval "$(conda shell.bash hook)"
    conda activate "$ENVIRONMENT_NAME"
  fi
  echo "Using interpreter at $(which python3)"
}

get_script_name() (
  # Get the name of the script.
  script_name=$(basename -- "$0")
  # Now strip off the extension.
  script_name="${script_name%.*}"
  echo $script_name
)

get_output_folder() (
  PROJECT_CACHE=$(get_project_cache)
  SCRIPT_NAME=$(get_script_name)
  OUTPUT_FOLDER="$PROJECT_CACHE/$SCRIPT_NAME"
  echo $OUTPUT_FOLDER
)

print_output_folder() (
  OUTPUT_FOLDER=$(get_output_folder)
  printf "Output folder: $OUTPUT_FOLDER\n"
)

make_output_folder() (
  OUTPUT_FOLDER=$(get_output_folder)
  # Warn if the output folder exists and is not empty.
  if [ -d "$OUTPUT_FOLDER" ]; then
    echo "WARNING: Output folder $OUTPUT_FOLDER already exists."
    if [ "$(ls -A $OUTPUT_FOLDER)" ]; then
      echo "WARNING: Output folder $OUTPUT_FOLDER is not empty."
    else
      echo "Output folder $OUTPUT_FOLDER is empty."
    fi
  else
    echo "Creating output folder $OUTPUT_FOLDER"
    mkdir -p $OUTPUT_FOLDER
  fi
)

get_num_gpus() (
  # Get the number of GPUs on the system.
  num_gpus=$(nvidia-smi -L | wc -l)
  echo $num_gpus
)

# Check if the script was run with the --summary flag. In this case, simply print
# out diagnostic information such as the conda environment, the output folder,
# and whether the output folder has anything in it or not.
exit_if_only_summary_needed() {
  args=("$@")
  # Check if any of the args are equal to --summary.
  for arg in "${args[@]}"; do
    if [ "$arg" == "--summary" ]; then
      echo "Running in summary mode."
      echo "Using interpreter at $(which python3)"
      echo "Output folder: $(get_output_folder)"
      if [ -d "$(get_output_folder)" ]; then
        output_folder="$(get_output_folder)"
        if [ "$(ls -A $output_folder)" ]; then
          echo "Output folder is not empty."
          # List the contents of output folder.
          echo "Contents of output folder:"
          ls $output_folder
        else
          echo "Output folder is empty."
        fi
      else
        echo "Output folder does not exist."
      fi
      exit 0
    fi
  done
}
