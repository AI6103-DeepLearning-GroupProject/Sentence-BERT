#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#
### Specify Memory allocate to this job ###
#SBATCH --mem=32G
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
### Specify Time Limit ###
#SBATCH --time=240
#
#SBATCH --job-name=sbert_eval_senteval
#SBATCH --output=result/logs/%x-%j.out
#SBATCH --error=result/logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/sbert024}"
PYTHON_MODULE="${PYTHON_MODULE:-python312}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8.0}"
ROOT_RUNTIME_CONFIG="${ROOT_RUNTIME_CONFIG:-config.yaml}"

EXPERIMENT="${EXPERIMENT:-}"
MODEL_VARIANT="${MODEL_VARIANT:-}"
DATA_MODE="${DATA_MODE:-}"
TRAIN_RESULT_DIR="${TRAIN_RESULT_DIR:-}"

MODEL_NAME="${MODEL_NAME:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

SENTEVAL_TOOLKIT_PATH="${SENTEVAL_TOOLKIT_PATH:-}"
SENTEVAL_DATA_PATH="${SENTEVAL_DATA_PATH:-}"
SENTEVAL_TASKS="${SENTEVAL_TASKS:-}"
SENTEVAL_KFOLD="${SENTEVAL_KFOLD:-}"
SENTEVAL_MAIN_METRIC="${SENTEVAL_MAIN_METRIC:-}"
SENTEVAL_CLASSIFIER_NHID="${SENTEVAL_CLASSIFIER_NHID:-}"
SENTEVAL_CLASSIFIER_OPTIM="${SENTEVAL_CLASSIFIER_OPTIM:-}"
SENTEVAL_CLASSIFIER_BATCH_SIZE="${SENTEVAL_CLASSIFIER_BATCH_SIZE:-}"
SENTEVAL_CLASSIFIER_TENACITY="${SENTEVAL_CLASSIFIER_TENACITY:-}"
SENTEVAL_CLASSIFIER_EPOCH_SIZE="${SENTEVAL_CLASSIFIER_EPOCH_SIZE:-}"

BATCH_SIZE="${BATCH_SIZE:-}"
SEED="${SEED:-}"
NUM_RUNS="${NUM_RUNS:-}"
SEED_STEP="${SEED_STEP:-}"
DATA_AUTO_DOWNLOAD="${DATA_AUTO_DOWNLOAD:-}"

module purge
module load "${PYTHON_MODULE}"
module load "${CUDA_MODULE}"

cd "${PROJECT_DIR}"
mkdir -p result/logs
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

if [ -z "${EXPERIMENT}" ]; then
  EXPERIMENT="$(read_root_cfg "slurm.train.experiment" "runtime")"
fi
if [ -z "${MODEL_VARIANT}" ]; then
  MODEL_VARIANT="$(read_root_cfg "slurm.train.model_variant" "sbert_base")"
fi
if [ -z "${DATA_MODE}" ]; then
  DATA_MODE="$(read_root_cfg "slurm.train.data_mode" "sts")"
fi
if [ -z "${TRAIN_RESULT_DIR}" ]; then
  TRAIN_RESULT_DIR="$(read_root_cfg "slurm.train.result_dir" "")"
fi

if [ -z "${SENTEVAL_TOOLKIT_PATH}" ]; then
  SENTEVAL_TOOLKIT_PATH="$(read_root_cfg "slurm.eval.senteval_toolkit_path" "datasets/SentEval")"
fi
if [ -z "${SENTEVAL_DATA_PATH}" ]; then
  SENTEVAL_DATA_PATH="$(read_root_cfg "slurm.eval.senteval_data_path" "datasets/SentEval/data/downstream")"
fi
if [ -z "${SENTEVAL_TASKS}" ]; then
  SENTEVAL_TASKS="$(read_root_cfg "slurm.eval.senteval_tasks" "MR,CR,SUBJ,MPQA,SST2,TREC,MRPC")"
fi
if [ -z "${SENTEVAL_KFOLD}" ]; then
  SENTEVAL_KFOLD="$(read_root_cfg "slurm.eval.senteval_kfold" "2")"
fi
if [ -z "${SENTEVAL_MAIN_METRIC}" ]; then
  SENTEVAL_MAIN_METRIC="$(read_root_cfg "slurm.eval.senteval_main_metric" "auto")"
fi
if [ -z "${SENTEVAL_CLASSIFIER_NHID}" ]; then
  SENTEVAL_CLASSIFIER_NHID="$(read_root_cfg "slurm.eval.senteval_classifier_nhid" "0")"
fi
if [ -z "${SENTEVAL_CLASSIFIER_OPTIM}" ]; then
  SENTEVAL_CLASSIFIER_OPTIM="$(read_root_cfg "slurm.eval.senteval_classifier_optim" "rmsprop")"
fi
if [ -z "${SENTEVAL_CLASSIFIER_BATCH_SIZE}" ]; then
  SENTEVAL_CLASSIFIER_BATCH_SIZE="$(read_root_cfg "slurm.eval.senteval_classifier_batch_size" "128")"
fi
if [ -z "${SENTEVAL_CLASSIFIER_TENACITY}" ]; then
  SENTEVAL_CLASSIFIER_TENACITY="$(read_root_cfg "slurm.eval.senteval_classifier_tenacity" "3")"
fi
if [ -z "${SENTEVAL_CLASSIFIER_EPOCH_SIZE}" ]; then
  SENTEVAL_CLASSIFIER_EPOCH_SIZE="$(read_root_cfg "slurm.eval.senteval_classifier_epoch_size" "2")"
fi
if [ -z "${BATCH_SIZE}" ]; then
  BATCH_SIZE="$(read_root_cfg "slurm.eval.batch_size" "64")"
fi
if [ -z "${SEED}" ]; then
  SEED="$(read_root_cfg "slurm.eval.seed" "42")"
fi
if [ -z "${NUM_RUNS}" ]; then
  NUM_RUNS="$(read_root_cfg "slurm.eval.num_runs" "1")"
fi
if [ -z "${SEED_STEP}" ]; then
  SEED_STEP="$(read_root_cfg "slurm.eval.seed_step" "1")"
fi
if [ -z "${DATA_AUTO_DOWNLOAD}" ]; then
  DATA_AUTO_DOWNLOAD="$(read_root_cfg "slurm.eval.data_auto_download" "1")"
fi

if [ -z "${TRAIN_RESULT_DIR}" ]; then
  DATA_MODE_NORM="$(echo "${DATA_MODE}" | tr '[:upper:]' '[:lower:]' | tr '+-' '__')"
  if [ "${EXPERIMENT}" = "runtime" ]; then
    TRAIN_RESULT_DIR="result/${EXPERIMENT}_${MODEL_VARIANT}_${DATA_MODE_NORM}"
  else
    TRAIN_RESULT_DIR="result/${EXPERIMENT}"
  fi
fi

if [ -z "${MODEL_NAME}" ]; then
  MODEL_NAME="${TRAIN_RESULT_DIR}/model"
fi

if [ -z "${OUTPUT_DIR}" ]; then
  OUTPUT_DIR="${TRAIN_RESULT_DIR}/eval_senteval"
fi

if ! [[ "${SENTEVAL_KFOLD}" =~ ^[0-9]+$ ]] || [ "${SENTEVAL_KFOLD}" -lt 1 ]; then
  echo "SENTEVAL_KFOLD must be an integer >= 1. Got: ${SENTEVAL_KFOLD}" >&2
  exit 1
fi
if ! [[ "${SENTEVAL_CLASSIFIER_BATCH_SIZE}" =~ ^[0-9]+$ ]] || [ "${SENTEVAL_CLASSIFIER_BATCH_SIZE}" -lt 1 ]; then
  echo "SENTEVAL_CLASSIFIER_BATCH_SIZE must be an integer >= 1. Got: ${SENTEVAL_CLASSIFIER_BATCH_SIZE}" >&2
  exit 1
fi
if ! [[ "${SENTEVAL_CLASSIFIER_TENACITY}" =~ ^[0-9]+$ ]] || [ "${SENTEVAL_CLASSIFIER_TENACITY}" -lt 1 ]; then
  echo "SENTEVAL_CLASSIFIER_TENACITY must be an integer >= 1. Got: ${SENTEVAL_CLASSIFIER_TENACITY}" >&2
  exit 1
fi
if ! [[ "${SENTEVAL_CLASSIFIER_EPOCH_SIZE}" =~ ^[0-9]+$ ]] || [ "${SENTEVAL_CLASSIFIER_EPOCH_SIZE}" -lt 1 ]; then
  echo "SENTEVAL_CLASSIFIER_EPOCH_SIZE must be an integer >= 1. Got: ${SENTEVAL_CLASSIFIER_EPOCH_SIZE}" >&2
  exit 1
fi
if ! [[ "${SEED}" =~ ^-?[0-9]+$ ]]; then
  echo "SEED must be an integer. Got: ${SEED}" >&2
  exit 1
fi
if ! [[ "${NUM_RUNS}" =~ ^[0-9]+$ ]] || [ "${NUM_RUNS}" -lt 1 ]; then
  echo "NUM_RUNS must be an integer >= 1. Got: ${NUM_RUNS}" >&2
  exit 1
fi
if ! [[ "${SEED_STEP}" =~ ^[0-9]+$ ]] || [ "${SEED_STEP}" -lt 1 ]; then
  echo "SEED_STEP must be an integer >= 1. Got: ${SEED_STEP}" >&2
  exit 1
fi

if [ ! -d "${MODEL_NAME}" ] || [ -z "$(ls -A "${MODEL_NAME}" 2>/dev/null)" ]; then
  echo "Model directory is missing or empty: ${MODEL_NAME}" >&2
  echo "Run training first (e.g., sbatch scripts/02_train_sbert.slurm)." >&2
  exit 1
fi

if [ ! -d "${SENTEVAL_TOOLKIT_PATH}" ] || [ ! -d "${SENTEVAL_DATA_PATH}" ]; then
  if [ "${DATA_AUTO_DOWNLOAD}" = "1" ]; then
    echo "SentEval assets are missing."
    echo "Please run: sbatch scripts/00_download_senteval.slurm" >&2
  fi
  if [ ! -d "${SENTEVAL_TOOLKIT_PATH}" ]; then
    echo "Missing SentEval toolkit directory: ${SENTEVAL_TOOLKIT_PATH}" >&2
  fi
  if [ ! -d "${SENTEVAL_DATA_PATH}" ]; then
    echo "Missing SentEval data directory: ${SENTEVAL_DATA_PATH}" >&2
  fi
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

SAFE_MODEL_NAME="${MODEL_NAME//\//_}"
LOG_FILE="result/logs/eval_senteval_${SAFE_MODEL_NAME}-${SLURM_JOB_ID}.log"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "Model: ${MODEL_NAME}" | tee "${LOG_FILE}"
echo "Output: ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "SentEval tasks: ${SENTEVAL_TASKS}" | tee -a "${LOG_FILE}"

python examples/evaluate_senteval.py \
  --model-name "${MODEL_NAME}" \
  --senteval-toolkit-path "${SENTEVAL_TOOLKIT_PATH}" \
  --senteval-data-path "${SENTEVAL_DATA_PATH}" \
  --tasks "${SENTEVAL_TASKS}" \
  --batch-size "${BATCH_SIZE}" \
  --kfold "${SENTEVAL_KFOLD}" \
  --main-metric "${SENTEVAL_MAIN_METRIC}" \
  --classifier-nhid "${SENTEVAL_CLASSIFIER_NHID}" \
  --classifier-optim "${SENTEVAL_CLASSIFIER_OPTIM}" \
  --classifier-batch-size "${SENTEVAL_CLASSIFIER_BATCH_SIZE}" \
  --classifier-tenacity "${SENTEVAL_CLASSIFIER_TENACITY}" \
  --classifier-epoch-size "${SENTEVAL_CLASSIFIER_EPOCH_SIZE}" \
  --seed "${SEED}" \
  --num-runs "${NUM_RUNS}" \
  --seed-step "${SEED_STEP}" \
  --output-path "${OUTPUT_DIR}" 2>&1 | tee -a "${LOG_FILE}"

echo "SentEval evaluation finished. Results in ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
