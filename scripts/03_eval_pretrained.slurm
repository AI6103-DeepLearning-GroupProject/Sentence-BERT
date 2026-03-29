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
#SBATCH --time=320
#
#SBATCH --job-name=sbert_eval
#SBATCH --output=output/slurm/%x-%j.out
#SBATCH --error=output/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/sbert024}"
PYTHON_MODULE="${PYTHON_MODULE:-python312}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.8.0}"

MODEL_NAME="${MODEL_NAME:-bert-base-nli-mean-tokens}"
STS_PATH="${STS_PATH:-datasets/stsbenchmark}"
STS_TEST_FILE="${STS_TEST_FILE:-sts-test.csv}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-42}"
NUM_RUNS="${NUM_RUNS:-10}"
SEED_STEP="${SEED_STEP:-1}"
MAIN_SIMILARITY="${MAIN_SIMILARITY:-cosine}"
SAFE_MODEL_NAME="${MODEL_NAME//\//_}"
OUTPUT_DIR="${OUTPUT_DIR:-output/eval_pretrained_${SAFE_MODEL_NAME}_${SLURM_JOB_ID}}"
DATA_AUTO_DOWNLOAD="${DATA_AUTO_DOWNLOAD:-1}"

module purge
module load "${PYTHON_MODULE}"
module load "${CUDA_MODULE}"

cd "${PROJECT_DIR}"
mkdir -p output/slurm "${OUTPUT_DIR}"
source "${VENV_DIR}/bin/activate"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

LOG_FILE="output/slurm/eval_${SAFE_MODEL_NAME}-${SLURM_JOB_ID}.log"
STS_TEST_PATH="${STS_PATH}/${STS_TEST_FILE}"

sync_stsbenchmark() {
  local source_base src_dir file_name
  for source_base in datasets dataset examples/datasets experiment/datasets; do
    src_dir="${source_base}/stsbenchmark"
    if [ -d "${src_dir}" ] && [ "${src_dir}" != "${STS_PATH}" ]; then
      mkdir -p "${STS_PATH}"
      cp -a "${src_dir}/." "${STS_PATH}/"
    fi

    if [ -f "${source_base}/sts-train.csv" ] || [ -f "${source_base}/sts-dev.csv" ] || [ -f "${source_base}/sts-test.csv" ]; then
      mkdir -p "${STS_PATH}"
      for file_name in sts-train.csv sts-dev.csv sts-test.csv sts-train-dev.csv correlation.pl readme.txt Readme.txt README.md LICENSE.txt; do
        if [ -f "${source_base}/${file_name}" ]; then
          cp -a "${source_base}/${file_name}" "${STS_PATH}/"
        fi
      done
    fi
  done
}

sync_stsbenchmark

if [ ! -f "${STS_TEST_PATH}" ] && [ "${DATA_AUTO_DOWNLOAD}" = "1" ]; then
  echo "Missing STS test file. Trying automatic download..."
  python examples/datasets/get_data.py
  sync_stsbenchmark
fi

if [ ! -f "${STS_TEST_PATH}" ]; then
  echo "Missing required STS test file: ${STS_TEST_PATH}" >&2
  echo "Run: sbatch scripts/01_setup_env_data.slurm" >&2
  echo "Or submit this job with DATA_AUTO_DOWNLOAD=1 (default)." >&2
  exit 1
fi

python examples/evaluate_pretrained.py \
  --model-name "${MODEL_NAME}" \
  --sts-path "${STS_PATH}" \
  --sts-test-file "${STS_TEST_FILE}" \
  --batch-size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --num-runs "${NUM_RUNS}" \
  --seed-step "${SEED_STEP}" \
  --main-similarity "${MAIN_SIMILARITY}" \
  --output-path "${OUTPUT_DIR}" 2>&1 | tee "${LOG_FILE}"
