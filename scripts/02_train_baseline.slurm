#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#
### Specify Memory allocate to this job ###
#SBATCH --mem=40G
#
### Specify number of core (CPU) to allocate to per task ###
#SBATCH --cpus-per-task=1
#
### Specify number of node to compute ###
#SBATCH --nodes=1
#
### Optional: Specify node to execute the job ###
### Remove 1st # at next line for the option to take effect ###
#SBATCH --nodelist=TC2N06
#
### Specify Time Limit, format: <min> or <min>:<sec> or <hr>:<min>:<sec> or <days>-<hr>:<min>:<sec> or <days>-<hr> ###
#SBATCH --time=360
#
#SBATCH --job-name=sbert_baseline
#SBATCH --output=output/slurm/%x-%j.out
#SBATCH --error=output/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/sbert024}"
PYTHON_MODULE="${PYTHON_MODULE:-python312}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8.0}"
ROOT_RUNTIME_CONFIG="${ROOT_RUNTIME_CONFIG:-config.yaml}"
CONFIG_PATH="${CONFIG_PATH:-}"
EXPERIMENT="${EXPERIMENT:-}"
MODEL_VARIANT="${MODEL_VARIANT:-}"
DATA_MODE="${DATA_MODE:-}"
EPOCHS="${EPOCHS:-}"
SEED="${SEED:-}"
DATA_AUTO_DOWNLOAD="${DATA_AUTO_DOWNLOAD:-}"

module purge
module load "${PYTHON_MODULE}"
module load "${CUDA_MODULE}"

cd "${PROJECT_DIR}"
mkdir -p output/slurm
source "${VENV_DIR}/bin/activate"

read_root_cfg() {
  local key="$1"
  local fallback="$2"
  python - "$ROOT_RUNTIME_CONFIG" "$key" "$fallback" <<'PY'
import os
import sys
import yaml

cfg_path, dotted_key, fallback = sys.argv[1], sys.argv[2], sys.argv[3]
value = fallback

if os.path.isfile(cfg_path):
    with open(cfg_path, encoding="utf-8") as f_in:
        loaded = yaml.safe_load(f_in) or {}
    node = loaded
    for part in dotted_key.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            node = None
            break
    if node is not None:
        value = node

if isinstance(value, bool):
    value = "1" if value else "0"
print(value)
PY
}

if [ -z "${CONFIG_PATH}" ]; then
  CONFIG_PATH="$(read_root_cfg "slurm.train.config_path" "examples/config.yaml")"
fi
if [ -z "${EXPERIMENT}" ]; then
  EXPERIMENT="$(read_root_cfg "slurm.train.experiment" "runtime")"
fi
if [ -z "${MODEL_VARIANT}" ]; then
  MODEL_VARIANT="$(read_root_cfg "slurm.train.model_variant" "sbert_base")"
fi
if [ -z "${DATA_MODE}" ]; then
  DATA_MODE="$(read_root_cfg "slurm.train.data_mode" "sts")"
fi
if [ -z "${EPOCHS}" ]; then
  EPOCHS="$(read_root_cfg "slurm.train.epochs" "4")"
fi
if [ -z "${SEED}" ]; then
  SEED="$(read_root_cfg "slurm.train.seed" "42")"
fi
if [ -z "${DATA_AUTO_DOWNLOAD}" ]; then
  DATA_AUTO_DOWNLOAD="$(read_root_cfg "slurm.train.data_auto_download" "1")"
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

DATA_MODE_NORM="$(echo "${DATA_MODE}" | tr '[:upper:]' '[:lower:]' | tr '+-' '__')"
if [ "${EXPERIMENT}" = "runtime" ]; then
  LOG_FILE="output/slurm/train_${EXPERIMENT}_${MODEL_VARIANT}_${DATA_MODE_NORM}-${SLURM_JOB_ID}.log"
else
  LOG_FILE="output/slurm/train_${EXPERIMENT}-${SLURM_JOB_ID}.log"
fi

sync_known_datasets() {
  local source_base src_dir dst_dir
  for source_base in dataset examples/datasets experiment/datasets; do
    for name in AllNLI stsbenchmark wikipedia-sections-triplets; do
      src_dir="${source_base}/${name}"
      dst_dir="datasets/${name}"
      if [ -d "${src_dir}" ]; then
        mkdir -p "${dst_dir}"
        cp -a "${src_dir}/." "${dst_dir}/"
      fi
    done
  done
}

normalize_flat_datasets() {
  local source_base file_name
  for source_base in datasets dataset examples/datasets experiment/datasets; do
    if [ -f "${source_base}/s1.train.gz" ] || [ -f "${source_base}/labels.train.gz" ]; then
      mkdir -p datasets/AllNLI
      for file_name in s1.train.gz s2.train.gz labels.train.gz s1.dev.gz s2.dev.gz labels.dev.gz README.md; do
        if [ -f "${source_base}/${file_name}" ]; then
          cp -a "${source_base}/${file_name}" "datasets/AllNLI/"
        fi
      done
    fi

    if [ -f "${source_base}/sts-train.csv" ] || [ -f "${source_base}/sts-dev.csv" ] || [ -f "${source_base}/sts-test.csv" ]; then
      mkdir -p datasets/stsbenchmark
      for file_name in sts-train.csv sts-dev.csv sts-test.csv sts-train-dev.csv correlation.pl readme.txt Readme.txt README.md LICENSE.txt; do
        if [ -f "${source_base}/${file_name}" ]; then
          cp -a "${source_base}/${file_name}" "datasets/stsbenchmark/"
        fi
      done
    fi

    if [ -f "${source_base}/train.csv" ] || [ -f "${source_base}/validation.csv" ] || [ -f "${source_base}/test.csv" ]; then
      mkdir -p datasets/wikipedia-sections-triplets
      for file_name in train.csv validation.csv test.csv readme.txt Readme.txt README.md LICENSE.txt; do
        if [ -f "${source_base}/${file_name}" ]; then
          cp -a "${source_base}/${file_name}" "datasets/wikipedia-sections-triplets/"
        fi
      done
    fi
  done
}

build_contrastive_pairs_if_needed() {
  if [ "${EXPERIMENT}" = "runtime" ] && [ "${DATA_MODE_NORM}" = "sts_nli_mnrl" ]; then
    local pairs_file="datasets/contrastive/pairs_train.tsv"
    if [ ! -f "${pairs_file}" ]; then
      if [ -f "datasets/AllNLI/s1.train.gz" ] && [ -f "datasets/AllNLI/s2.train.gz" ] && [ -f "datasets/AllNLI/labels.train.gz" ] && [ -f "datasets/stsbenchmark/sts-train.csv" ]; then
        echo "Building contrastive pairs for MNRL..."
        python examples/datasets/build_contrastive_pairs.py \
          --allnli-path datasets/AllNLI \
          --sts-path datasets/stsbenchmark \
          --output-path "${pairs_file}"
      else
        echo "Skip contrastive pair build for now: STS/NLI source files are not ready yet."
      fi
    fi
  fi
}

sync_known_datasets
normalize_flat_datasets
build_contrastive_pairs_if_needed

required_files=()
if [ "${EXPERIMENT}" = "runtime" ]; then
  case "${DATA_MODE_NORM}" in
    sts)
      required_files=(
        "datasets/stsbenchmark/sts-train.csv"
        "datasets/stsbenchmark/sts-dev.csv"
        "datasets/stsbenchmark/sts-test.csv"
      )
      ;;
    nli)
      required_files=(
        "datasets/AllNLI/s1.train.gz"
        "datasets/AllNLI/s2.train.gz"
        "datasets/AllNLI/labels.train.gz"
        "datasets/stsbenchmark/sts-dev.csv"
        "datasets/stsbenchmark/sts-test.csv"
      )
      ;;
    sts_nli)
      required_files=(
        "datasets/AllNLI/s1.train.gz"
        "datasets/AllNLI/s2.train.gz"
        "datasets/AllNLI/labels.train.gz"
        "datasets/stsbenchmark/sts-train.csv"
        "datasets/stsbenchmark/sts-dev.csv"
        "datasets/stsbenchmark/sts-test.csv"
      )
      ;;
    sts_nli_mnrl)
      required_files=(
        "datasets/AllNLI/s1.train.gz"
        "datasets/AllNLI/s2.train.gz"
        "datasets/AllNLI/labels.train.gz"
        "datasets/stsbenchmark/sts-train.csv"
        "datasets/stsbenchmark/sts-dev.csv"
        "datasets/stsbenchmark/sts-test.csv"
        "datasets/contrastive/pairs_train.tsv"
      )
      ;;
    *)
      echo "Unknown DATA_MODE='${DATA_MODE}'. Use sts, nli, sts+nli (sts_nli), or sts+nli+mnrl (sts_nli_mnrl)." >&2
      exit 1
      ;;
  esac
else
  case "${EXPERIMENT}" in
    nli_bert)
      required_files=(
        "datasets/AllNLI/s1.train.gz"
        "datasets/AllNLI/s2.train.gz"
        "datasets/AllNLI/labels.train.gz"
        "datasets/stsbenchmark/sts-dev.csv"
        "datasets/stsbenchmark/sts-test.csv"
      )
      ;;
    sts_bert|sts_continue_training_bert)
      required_files=(
        "datasets/stsbenchmark/sts-train.csv"
        "datasets/stsbenchmark/sts-dev.csv"
        "datasets/stsbenchmark/sts-test.csv"
      )
      ;;
    wikipedia_triplet_bert)
      required_files=(
        "datasets/wikipedia-sections-triplets/train.csv"
        "datasets/wikipedia-sections-triplets/validation.csv"
        "datasets/wikipedia-sections-triplets/test.csv"
      )
      ;;
    *)
      echo "Unknown EXPERIMENT='${EXPERIMENT}'. Update scripts/02_train_baseline.slurm required_files mapping." >&2
      exit 1
      ;;
  esac
fi

missing_data=0
for file_path in "${required_files[@]}"; do
  if [ ! -f "${file_path}" ]; then
    missing_data=1
    break
  fi
done

if [ "${missing_data}" -eq 1 ] && [ "${DATA_AUTO_DOWNLOAD}" = "1" ]; then
  echo "Required dataset files are missing. Trying automatic download..."
  python examples/datasets/get_data.py
  sync_known_datasets
  normalize_flat_datasets
  build_contrastive_pairs_if_needed
fi

for file_path in "${required_files[@]}"; do
  if [ ! -f "${file_path}" ]; then
    echo "Missing required dataset file: ${file_path}" >&2
    echo "Run: sbatch scripts/01_setup_env_data.slurm" >&2
    echo "Or submit this job with DATA_AUTO_DOWNLOAD=1 (default)." >&2
    exit 1
  fi
done

python examples/train_with_config.py \
  --config "${CONFIG_PATH}" \
  --experiment "${EXPERIMENT}" \
  --model-variant "${MODEL_VARIANT}" \
  --data-mode "${DATA_MODE}" \
  --epochs "${EPOCHS}" \
  --seed "${SEED}" 2>&1 | tee "${LOG_FILE}"
