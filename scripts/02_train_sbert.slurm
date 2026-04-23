#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#
### Specify Memory allocate to this job ###
#SBATCH --mem=44G
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
#SBATCH --job-name=train_SBERT
#SBATCH --output=result/logs/%x-%j.out
#SBATCH --error=result/logs/%x-%j.err

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
NUM_RUNS="${NUM_RUNS:-}"
SEED_STEP="${SEED_STEP:-}"
RESUME_STAGE="${RESUME_STAGE:-}"
RESUME_FROM="${RESUME_FROM:-}"
RESULT_DIR="${RESULT_DIR:-}"
MODEL_DIR="${MODEL_DIR:-}"

EVAL_TASK="${EVAL_TASK:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
DATA_AUTO_DOWNLOAD="${DATA_AUTO_DOWNLOAD:-}"

STS_PATH="${STS_PATH:-}"
STS_TEST_FILE="${STS_TEST_FILE:-}"
MAIN_SIMILARITY="${MAIN_SIMILARITY:-}"

TRIPLET_PATH="${TRIPLET_PATH:-}"
TRIPLET_TEST_FILE="${TRIPLET_TEST_FILE:-}"
MAIN_DISTANCE="${MAIN_DISTANCE:-}"
TRIPLET_S1_COL_IDX="${TRIPLET_S1_COL_IDX:-}"
TRIPLET_S2_COL_IDX="${TRIPLET_S2_COL_IDX:-}"
TRIPLET_S3_COL_IDX="${TRIPLET_S3_COL_IDX:-}"
TRIPLET_DELIMITER="${TRIPLET_DELIMITER:-}"
TRIPLET_QUOTING="${TRIPLET_QUOTING:-}"
TRIPLET_HAS_HEADER="${TRIPLET_HAS_HEADER:-}"

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
if [ -z "${NUM_RUNS}" ]; then
  NUM_RUNS="$(read_root_cfg "slurm.train.num_runs" "3")"
fi
if [ -z "${SEED_STEP}" ]; then
  SEED_STEP="$(read_root_cfg "slurm.train.seed_step" "1")"
fi
if [ -z "${RESUME_STAGE}" ]; then
  RESUME_STAGE="$(read_root_cfg "slurm.train.resume_stage" "1")"
fi
if [ -z "${RESUME_FROM}" ]; then
  RESUME_FROM="$(read_root_cfg "slurm.train.resume_from" "")"
fi
if [ -z "${RESULT_DIR}" ]; then
  RESULT_DIR="$(read_root_cfg "slurm.train.result_dir" "")"
fi
if [ -z "${EVAL_TASK}" ]; then
  EVAL_TASK="$(read_root_cfg "slurm.eval.task_type" "auto")"
fi
if [ -z "${BATCH_SIZE}" ]; then
  BATCH_SIZE="$(read_root_cfg "slurm.eval.batch_size" "32")"
fi
if [ -z "${STS_PATH}" ]; then
  STS_PATH="$(read_root_cfg "slurm.eval.sts_path" "datasets/stsbenchmark")"
fi
if [ -z "${STS_TEST_FILE}" ]; then
  STS_TEST_FILE="$(read_root_cfg "slurm.eval.sts_test_file" "sts-test.csv")"
fi
if [ -z "${MAIN_SIMILARITY}" ]; then
  MAIN_SIMILARITY="$(read_root_cfg "slurm.eval.main_similarity" "cosine")"
fi
if [ -z "${TRIPLET_PATH}" ]; then
  TRIPLET_PATH="$(read_root_cfg "slurm.eval.triplet_path" "datasets/wikipedia-sections-triplets")"
fi
if [ -z "${TRIPLET_TEST_FILE}" ]; then
  TRIPLET_TEST_FILE="$(read_root_cfg "slurm.eval.triplet_test_file" "test.csv")"
fi
if [ -z "${MAIN_DISTANCE}" ]; then
  MAIN_DISTANCE="$(read_root_cfg "slurm.eval.main_distance" "cosine")"
fi
if [ -z "${TRIPLET_S1_COL_IDX}" ]; then
  TRIPLET_S1_COL_IDX="$(read_root_cfg "slurm.eval.triplet_s1_col_idx" "1")"
fi
if [ -z "${TRIPLET_S2_COL_IDX}" ]; then
  TRIPLET_S2_COL_IDX="$(read_root_cfg "slurm.eval.triplet_s2_col_idx" "2")"
fi
if [ -z "${TRIPLET_S3_COL_IDX}" ]; then
  TRIPLET_S3_COL_IDX="$(read_root_cfg "slurm.eval.triplet_s3_col_idx" "3")"
fi
if [ -z "${TRIPLET_DELIMITER}" ]; then
  TRIPLET_DELIMITER="$(read_root_cfg "slurm.eval.triplet_delimiter" ",")"
fi
if [ -z "${TRIPLET_QUOTING}" ]; then
  TRIPLET_QUOTING="$(read_root_cfg "slurm.eval.triplet_quoting" "QUOTE_MINIMAL")"
fi
if [ -z "${TRIPLET_HAS_HEADER}" ]; then
  TRIPLET_HAS_HEADER="$(read_root_cfg "slurm.eval.triplet_has_header" "1")"
fi
if [ -z "${SENTEVAL_TOOLKIT_PATH}" ]; then
  SENTEVAL_TOOLKIT_PATH="$(read_root_cfg "slurm.eval.senteval_toolkit_path" "datasets/SentEval")"
fi
if [ -z "${SENTEVAL_DATA_PATH}" ]; then
  SENTEVAL_DATA_PATH="$(read_root_cfg "slurm.eval.senteval_data_path" "datasets/SentEval/data")"
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
if [ -z "${DATA_AUTO_DOWNLOAD}" ]; then
  DATA_AUTO_DOWNLOAD="$(read_root_cfg "slurm.train.data_auto_download" "")"
  if [ -z "${DATA_AUTO_DOWNLOAD}" ]; then
    DATA_AUTO_DOWNLOAD="$(read_root_cfg "slurm.eval.data_auto_download" "1")"
  fi
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
if ! [[ "${RESUME_STAGE}" =~ ^[0-9]+$ ]] || [ "${RESUME_STAGE}" -lt 1 ]; then
  echo "RESUME_STAGE must be an integer >= 1. Got: ${RESUME_STAGE}" >&2
  exit 1
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

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

DATA_MODE_NORM="$(echo "${DATA_MODE}" | tr '[:upper:]' '[:lower:]' | tr '+-' '__')"

if [ "${EVAL_TASK}" = "auto" ]; then
  if [ "${EXPERIMENT}" = "wikipedia_triplet_bert" ] || [ "${DATA_MODE_NORM}" = "wikipedia_triplet" ] || [ "${DATA_MODE_NORM}" = "wikipedia_triplet_mnrl" ]; then
    EVAL_TASK="triplet"
  else
    EVAL_TASK="sts"
  fi
fi

if [ "${EVAL_TASK}" != "sts" ] && [ "${EVAL_TASK}" != "triplet" ] && [ "${EVAL_TASK}" != "senteval" ]; then
  echo "EVAL_TASK must be one of: sts, triplet, senteval, auto. Got: ${EVAL_TASK}" >&2
  exit 1
fi

if [ -z "${RESULT_DIR}" ]; then
  if [ "${EXPERIMENT}" = "runtime" ]; then
    RESULT_DIR="result/${EXPERIMENT}_${MODEL_VARIANT}_${DATA_MODE_NORM}"
  else
    RESULT_DIR="result/${EXPERIMENT}"
  fi
fi

if [ -z "${MODEL_DIR}" ]; then
  MODEL_DIR="${RESULT_DIR}/model"
fi
EVAL_TMP_DIR="${RESULT_DIR}/eval_current"
RUNS_JSONL="${RESULT_DIR}/runs.jsonl"
SUMMARY_JSON="${RESULT_DIR}/summary.json"
SUMMARY_CSV="${RESULT_DIR}/summary_table.csv"

if [ "${EXPERIMENT}" = "runtime" ]; then
  LOG_FILE="result/logs/${EXPERIMENT}_${MODEL_VARIANT}_${DATA_MODE_NORM}-${SLURM_JOB_ID}.log"
else
  LOG_FILE="result/logs/${EXPERIMENT}-${SLURM_JOB_ID}.log"
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
  if [ "${EXPERIMENT}" = "runtime" ] && { [ "${DATA_MODE_NORM}" = "sts_nli_mnrl" ] || [ "${DATA_MODE_NORM}" = "sts_nli_mnrl_aoe" ]; }; then
    local sts_pairs_file="datasets/contrastive/pairs_train.tsv"
    local rebuild_pairs=0
    if [ ! -s "${sts_pairs_file}" ]; then
      rebuild_pairs=1
    elif ! awk -F '\t' 'NR == 1 { exit (NF >= 3 ? 0 : 1) }' "${sts_pairs_file}"; then
      echo "Existing MNRL pair file is legacy two-column format; rebuilding with hard negatives..."
      rebuild_pairs=1
    fi
    if [ "${rebuild_pairs}" = "1" ]; then
      if [ -f "datasets/AllNLI/s1.train.gz" ] && [ -f "datasets/AllNLI/s2.train.gz" ] && [ -f "datasets/AllNLI/labels.train.gz" ]; then
        echo "Building NLI entailment/contradiction hard-negative examples for MNRL..."
        python examples/datasets/build_contrastive_pairs.py \
          --allnli-path datasets/AllNLI \
          --sts-path datasets/stsbenchmark \
          --output-path "${sts_pairs_file}"
      else
        echo "Skip STS/NLI pair build for now: source files are not ready yet."
      fi
    fi
  fi

  if [ "${EXPERIMENT}" = "runtime" ] && [ "${DATA_MODE_NORM}" = "sts_nli_mnrl_aoe" ]; then
    local aoe_pairs_file="datasets/contrastive/aoe_pairs_train.tsv"
    if [ ! -f "${aoe_pairs_file}" ]; then
      if [ -f "datasets/AllNLI/s1.train.gz" ] && [ -f "datasets/AllNLI/s2.train.gz" ] && [ -f "datasets/AllNLI/labels.train.gz" ] && [ -f "datasets/stsbenchmark/sts-train.csv" ]; then
        echo "Building AoE supervised pairs from NLI+STS..."
        python examples/datasets/build_aoe_pairs.py \
          --allnli-path datasets/AllNLI \
          --sts-path datasets/stsbenchmark \
          --output-path "${aoe_pairs_file}"
      else
        echo "Skip AoE pair build for now: source files are not ready yet."
      fi
    fi
  fi

  if [ "${EXPERIMENT}" = "runtime" ] && [ "${DATA_MODE_NORM}" = "nli_mnrl" ]; then
    local nli_pairs_file="datasets/contrastive/nli_pairs_train.tsv"
    local rebuild_nli_pairs=0
    if [ ! -s "${nli_pairs_file}" ]; then
      rebuild_nli_pairs=1
    elif ! awk -F '\t' 'NR == 1 { exit (NF >= 3 ? 0 : 1) }' "${nli_pairs_file}"; then
      echo "Existing NLI MNRL pair file is legacy two-column format; rebuilding with hard negatives..."
      rebuild_nli_pairs=1
    fi
    if [ "${rebuild_nli_pairs}" = "1" ]; then
      if [ -f "datasets/AllNLI/s1.train.gz" ] && [ -f "datasets/AllNLI/s2.train.gz" ] && [ -f "datasets/AllNLI/labels.train.gz" ]; then
        echo "Building NLI entailment/contradiction hard-negative examples for MNRL..."
        python examples/datasets/build_contrastive_pairs.py \
          --allnli-path datasets/AllNLI \
          --output-path "${nli_pairs_file}" \
          --disable-sts
      else
        echo "Skip NLI-only pair build for now: AllNLI source files are not ready yet."
      fi
    fi
  fi

  if [ "${EXPERIMENT}" = "runtime" ] && [ "${DATA_MODE_NORM}" = "wikipedia_triplet_mnrl" ]; then
    local wiki_pairs_file="datasets/contrastive/wikipedia_pairs_train.tsv"
    local wiki_train_file="datasets/wikipedia-sections-triplets/train.csv"
    if [ ! -f "${wiki_pairs_file}" ]; then
      if [ -f "${wiki_train_file}" ]; then
        echo "Building wikipedia contrastive pairs for MNRL..."
        python examples/datasets/build_wikipedia_pairs.py \
          --triplet-train-file "${wiki_train_file}" \
          --output-path "${wiki_pairs_file}" \
          --s1-col-idx "${TRIPLET_S1_COL_IDX}" \
          --s2-col-idx "${TRIPLET_S2_COL_IDX}" \
          --delimiter "${TRIPLET_DELIMITER}" \
          --quoting "${TRIPLET_QUOTING}" \
          --has-header "${TRIPLET_HAS_HEADER}" \
          --bidirectional 0
      else
        echo "Skip wikipedia pair build for now: ${wiki_train_file} is not ready yet."
      fi
    fi
  fi
}

safe_cleanup_dir() {
  local target_dir="$1"
  local abs_target abs_root

  if [ ! -e "${target_dir}" ]; then
    return 0
  fi

  abs_target="$(cd "${target_dir}" && pwd -P)"
  abs_root="$(mkdir -p "${RESULT_DIR}" && cd "${RESULT_DIR}" && pwd -P)"

  case "${abs_target}" in
    "${abs_root}"/*)
      rm -rf "${target_dir}"
      ;;
    *)
      echo "Refusing to remove path outside RESULT_DIR: ${target_dir}" >&2
      exit 1
      ;;
  esac
}

run_and_log() {
  "$@" 2>&1 | tee -a "${LOG_FILE}"
  local cmd_status=${PIPESTATUS[0]}
  if [ "${cmd_status}" -ne 0 ]; then
    echo "Command failed (exit=${cmd_status}): $*" >&2
    exit "${cmd_status}"
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
    sts_nli_mnrl_aoe)
      required_files=(
        "datasets/AllNLI/s1.train.gz"
        "datasets/AllNLI/s2.train.gz"
        "datasets/AllNLI/labels.train.gz"
        "datasets/stsbenchmark/sts-train.csv"
        "datasets/stsbenchmark/sts-dev.csv"
        "datasets/stsbenchmark/sts-test.csv"
        "datasets/contrastive/pairs_train.tsv"
        "datasets/contrastive/aoe_pairs_train.tsv"
      )
      ;;
    nli_mnrl)
      required_files=(
        "datasets/AllNLI/s1.train.gz"
        "datasets/AllNLI/s2.train.gz"
        "datasets/AllNLI/labels.train.gz"
        "datasets/contrastive/nli_pairs_train.tsv"
      )
      ;;
    wikipedia_triplet)
      required_files=(
        "datasets/wikipedia-sections-triplets/train.csv"
        "datasets/wikipedia-sections-triplets/validation.csv"
        "datasets/wikipedia-sections-triplets/test.csv"
      )
      ;;
    wikipedia_triplet_mnrl)
      required_files=(
        "datasets/wikipedia-sections-triplets/train.csv"
        "datasets/wikipedia-sections-triplets/validation.csv"
        "datasets/wikipedia-sections-triplets/test.csv"
        "datasets/contrastive/wikipedia_pairs_train.tsv"
      )
      ;;
    *)
      echo "Unknown DATA_MODE='${DATA_MODE}'. Use sts, nli, nli_mnrl, sts_nli, sts_nli_mnrl, sts_nli_mnrl_aoe, wikipedia_triplet, or wikipedia_triplet_mnrl." >&2
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
      echo "Unknown EXPERIMENT='${EXPERIMENT}'. Update scripts/02_train_sbert.slurm required_files mapping." >&2
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

if [ "${EVAL_TASK}" = "sts" ]; then
  STS_TEST_PATH="${STS_PATH}/${STS_TEST_FILE}"
  if [ ! -f "${STS_TEST_PATH}" ]; then
    echo "Missing required STS test file for evaluation: ${STS_TEST_PATH}" >&2
    exit 1
  fi
elif [ "${EVAL_TASK}" = "triplet" ]; then
  TRIPLET_TEST_PATH="${TRIPLET_PATH}/${TRIPLET_TEST_FILE}"
  if [ ! -f "${TRIPLET_TEST_PATH}" ]; then
    echo "Missing required triplet test file for evaluation: ${TRIPLET_TEST_PATH}" >&2
    exit 1
  fi
else
  if [ ! -d "${SENTEVAL_TOOLKIT_PATH}" ]; then
    echo "Missing SentEval toolkit directory: ${SENTEVAL_TOOLKIT_PATH}" >&2
    exit 1
  fi
  if [ ! -d "${SENTEVAL_DATA_PATH}" ]; then
    echo "Missing SentEval data directory: ${SENTEVAL_DATA_PATH}" >&2
    exit 1
  fi
fi

mkdir -p "${RESULT_DIR}"
: > "${RUNS_JSONL}"

echo "Result directory: ${RESULT_DIR}" | tee -a "${LOG_FILE}"
echo "Model directory (overwritten each run): ${MODEL_DIR}" | tee -a "${LOG_FILE}"
echo "Runs: ${NUM_RUNS}, seed_start=${SEED}, seed_step=${SEED_STEP}" | tee -a "${LOG_FILE}"
echo "Resume stage: ${RESUME_STAGE}" | tee -a "${LOG_FILE}"
echo "Evaluation task: ${EVAL_TASK}" | tee -a "${LOG_FILE}"

for run_idx in $(seq 1 "${NUM_RUNS}"); do
  run_seed=$((SEED + (run_idx - 1) * SEED_STEP))
  echo "==== Run ${run_idx}/${NUM_RUNS} | seed=${run_seed} ====" | tee -a "${LOG_FILE}"

  run_resume_stage=1
  run_resume_from=""

  if [ "${run_idx}" -eq 1 ] && [ "${RESUME_STAGE}" -gt 1 ]; then
    run_resume_stage="${RESUME_STAGE}"
    if [ -n "${RESUME_FROM}" ]; then
      run_resume_from="${RESUME_FROM}"
    else
      run_resume_from="${MODEL_DIR}"
    fi

    if [ ! -d "${run_resume_from}" ] || [ -z "$(ls -A "${run_resume_from}" 2>/dev/null)" ]; then
      echo "Resume requested but model directory is missing/empty: ${run_resume_from}" >&2
      exit 1
    fi
    echo "Resume enabled for run 1: start_stage=${run_resume_stage}, resume_from=${run_resume_from}" | tee -a "${LOG_FILE}"
  else
    safe_cleanup_dir "${MODEL_DIR}"
    mkdir -p "${MODEL_DIR}"
  fi

  train_cmd=(
    python examples/train_with_config.py
    --config "${CONFIG_PATH}"
    --experiment "${EXPERIMENT}"
    --model-variant "${MODEL_VARIANT}"
    --data-mode "${DATA_MODE}"
    --epochs "${EPOCHS}"
    --seed "${run_seed}"
    --output-path "${MODEL_DIR}"
  )
  if [ "${run_resume_stage}" -gt 1 ]; then
    train_cmd+=(--resume-from "${run_resume_from}" --start-stage "${run_resume_stage}")
  fi
  run_and_log "${train_cmd[@]}"

  safe_cleanup_dir "${EVAL_TMP_DIR}"
  mkdir -p "${EVAL_TMP_DIR}"

  if [ "${EVAL_TASK}" = "sts" ]; then
    run_and_log python examples/evaluate_pretrained.py \
      --model-name "${MODEL_DIR}" \
      --sts-path "${STS_PATH}" \
      --sts-test-file "${STS_TEST_FILE}" \
      --batch-size "${BATCH_SIZE}" \
      --seed "${run_seed}" \
      --num-runs 1 \
      --seed-step 1 \
      --main-similarity "${MAIN_SIMILARITY}" \
      --output-path "${EVAL_TMP_DIR}"
  elif [ "${EVAL_TASK}" = "triplet" ]; then
    run_and_log python examples/evaluate_triplet_pretrained.py \
      --model-name "${MODEL_DIR}" \
      --triplet-path "${TRIPLET_PATH}" \
      --triplet-test-file "${TRIPLET_TEST_FILE}" \
      --batch-size "${BATCH_SIZE}" \
      --seed "${run_seed}" \
      --num-runs 1 \
      --seed-step 1 \
      --main-distance "${MAIN_DISTANCE}" \
      --s1-col-idx "${TRIPLET_S1_COL_IDX}" \
      --s2-col-idx "${TRIPLET_S2_COL_IDX}" \
      --s3-col-idx "${TRIPLET_S3_COL_IDX}" \
      --delimiter "${TRIPLET_DELIMITER}" \
      --quoting "${TRIPLET_QUOTING}" \
      --has-header "${TRIPLET_HAS_HEADER}" \
      --output-path "${EVAL_TMP_DIR}"
  else
    run_and_log python examples/evaluate_senteval.py \
      --model-name "${MODEL_DIR}" \
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
      --seed "${run_seed}" \
      --num-runs 1 \
      --seed-step 1 \
      --output-path "${EVAL_TMP_DIR}"
  fi

  if [ "${EVAL_TASK}" = "sts" ] && [ ! -f "${EVAL_TMP_DIR}/summary.json" ]; then
    echo "Primary STS evaluator produced no summary.json. Running fallback STS evaluator..." | tee -a "${LOG_FILE}"
    run_and_log python - "${MODEL_DIR}" "${STS_PATH}" "${STS_TEST_FILE}" "${BATCH_SIZE}" "${run_seed}" "${MAIN_SIMILARITY}" "${EVAL_TMP_DIR}" <<'PY'
import csv
import json
import os
import sys

from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer, SentencesDataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.evaluation import SimilarityFunction
from sentence_transformers.readers import STSDataReader
from sentence_transformers.util import set_seed


def resolve_main_similarity(name):
    if name == "cosine":
        return SimilarityFunction.COSINE
    if name == "euclidean":
        return SimilarityFunction.EUCLIDEAN
    if name == "manhattan":
        return SimilarityFunction.MANHATTAN
    if name == "dot":
        return SimilarityFunction.DOT_PRODUCT
    return None


def read_eval_metrics(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f_in:
        rows = list(csv.DictReader(f_in))
    if not rows:
        raise ValueError("No rows found in {}".format(csv_path))
    last_row = rows[-1]
    metrics = {}
    for key, value in last_row.items():
        if key in {"epoch", "steps"}:
            continue
        metrics[key] = float(value)
    return metrics


model_name = sys.argv[1]
sts_path = sys.argv[2]
sts_test_file = sys.argv[3]
batch_size = int(sys.argv[4])
seed = int(sys.argv[5])
main_similarity_name = sys.argv[6]
output_path = sys.argv[7]

os.makedirs(output_path, exist_ok=True)

set_seed(seed, deterministic=True)
model = SentenceTransformer(model_name)
sts_reader = STSDataReader(sts_path)
test_data = SentencesDataset(examples=sts_reader.get_examples(sts_test_file), model=model)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
evaluator = EmbeddingSimilarityEvaluator(
    test_dataloader,
    name="sts_test",
    main_similarity=resolve_main_similarity(main_similarity_name),
)

score = model.evaluate(evaluator, output_path=output_path)
metrics = read_eval_metrics(os.path.join(output_path, "similarity_evaluation_sts_test_results.csv"))

summary = {
    "model_name": model_name,
    "sts_path": sts_path,
    "sts_test_file": sts_test_file,
    "batch_size": batch_size,
    "main_similarity": main_similarity_name,
    "num_runs": 1,
    "seed_start": seed,
    "seed_step": 1,
    "seeds": [seed],
    "aggregate": {"score_mean": float(score), "score_std": 0.0},
    "runs": [
        {
            "run_index": 1,
            "seed": seed,
            "score": float(score),
            "metrics": metrics,
            "output_path": output_path,
        }
    ],
    "output_path": output_path,
}

for key, value in metrics.items():
    summary["aggregate"]["{}_mean".format(key)] = float(value)
    summary["aggregate"]["{}_std".format(key)] = 0.0

with open(os.path.join(output_path, "summary.json"), "w", encoding="utf-8") as f_out:
    json.dump(summary, f_out, indent=2)

with open(os.path.join(output_path, "summary_table.csv"), "w", newline="", encoding="utf-8") as f_out:
    fieldnames = ["run_index", "seed", "score"] + sorted(metrics.keys())
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()
    row = {"run_index": 1, "seed": seed, "score": float(score)}
    row.update(metrics)
    writer.writerow(row)
PY
  fi

  echo "Eval artifact dir: ${EVAL_TMP_DIR}" | tee -a "${LOG_FILE}"
  ls -la "${EVAL_TMP_DIR}" 2>&1 | tee -a "${LOG_FILE}" || true
  find "${RESULT_DIR}" -maxdepth 4 -name "summary.json" 2>/dev/null | tee -a "${LOG_FILE}" || true

  python - "${EVAL_TMP_DIR}/summary.json" "${run_idx}" >> "${RUNS_JSONL}" <<'PY'
import glob
import json
import os
import sys

summary_path = sys.argv[1].strip().strip("\"'“”‘’")
run_index = int(sys.argv[2])

if not os.path.isfile(summary_path):
    eval_dir = os.path.dirname(summary_path)
    candidates = sorted(glob.glob(os.path.join(eval_dir, "**", "summary.json"), recursive=True))
    if candidates:
        summary_path = candidates[0]
    else:
        raise FileNotFoundError(
            "Evaluation summary is missing: {}. Files under {}: {}".format(
                summary_path,
                eval_dir,
                sorted(os.listdir(eval_dir)) if os.path.isdir(eval_dir) else "<missing dir>",
            )
        )

with open(summary_path, encoding="utf-8") as f_in:
    summary = json.load(f_in)

run = summary["runs"][0]
row = {
    "run_index": run_index,
    "seed": int(run["seed"]),
    "score": float(run["score"]),
    "metrics": run["metrics"],
}
print(json.dumps(row))
PY
done

python - "${RUNS_JSONL}" "${SUMMARY_JSON}" "${SUMMARY_CSV}" "${MODEL_DIR}" "${NUM_RUNS}" "${SEED}" "${SEED_STEP}" "${EVAL_TASK}" "${MAIN_SIMILARITY}" "${STS_PATH}" "${STS_TEST_FILE}" "${MAIN_DISTANCE}" "${TRIPLET_PATH}" "${TRIPLET_TEST_FILE}" "${SENTEVAL_TOOLKIT_PATH}" "${SENTEVAL_DATA_PATH}" "${SENTEVAL_TASKS}" "${SENTEVAL_KFOLD}" "${SENTEVAL_MAIN_METRIC}" <<'PY'
import csv
import json
import statistics
import sys

runs_jsonl = sys.argv[1]
summary_json = sys.argv[2]
summary_csv = sys.argv[3]
model_dir = sys.argv[4]
num_runs = int(sys.argv[5])
seed_start = int(sys.argv[6])
seed_step = int(sys.argv[7])
eval_task = sys.argv[8]
main_similarity = sys.argv[9]
sts_path = sys.argv[10]
sts_test_file = sys.argv[11]
main_distance = sys.argv[12]
triplet_path = sys.argv[13]
triplet_test_file = sys.argv[14]
senteval_toolkit_path = sys.argv[15]
senteval_data_path = sys.argv[16]
senteval_tasks = sys.argv[17]
senteval_kfold = int(sys.argv[18])
senteval_main_metric = sys.argv[19]

runs = []
with open(runs_jsonl, encoding="utf-8") as f_in:
    for line in f_in:
        line = line.strip()
        if line:
            runs.append(json.loads(line))

if not runs:
    raise ValueError("No run records found in {}".format(runs_jsonl))

scores = [float(r["score"]) for r in runs]
aggregate = {
    "score_mean": float(statistics.mean(scores)),
    "score_std": float(statistics.stdev(scores)) if len(scores) > 1 else 0.0,
}

metric_keys = sorted(runs[0]["metrics"].keys())
for key in metric_keys:
    values = [float(r["metrics"][key]) for r in runs]
    aggregate["{}_mean".format(key)] = float(statistics.mean(values))
    aggregate["{}_std".format(key)] = float(statistics.stdev(values)) if len(values) > 1 else 0.0

summary = {
    "model_name": model_dir,
    "evaluation_task": eval_task,
    "num_runs": num_runs,
    "seed_start": seed_start,
    "seed_step": seed_step,
    "seeds": [int(r["seed"]) for r in runs],
    "aggregate": aggregate,
    "runs": runs,
}

if eval_task == "triplet":
    summary["main_distance"] = main_distance
    summary["triplet_path"] = triplet_path
    summary["triplet_test_file"] = triplet_test_file
elif eval_task == "senteval":
    summary["senteval_toolkit_path"] = senteval_toolkit_path
    summary["senteval_data_path"] = senteval_data_path
    summary["senteval_tasks"] = [task.strip() for task in senteval_tasks.split(",") if task.strip()]
    summary["senteval_kfold"] = senteval_kfold
    summary["senteval_main_metric"] = senteval_main_metric
else:
    summary["main_similarity"] = main_similarity
    summary["sts_path"] = sts_path
    summary["sts_test_file"] = sts_test_file

with open(summary_json, "w", encoding="utf-8") as f_out:
    json.dump(summary, f_out, indent=2)

with open(summary_csv, "w", newline="", encoding="utf-8") as f_out:
    fieldnames = ["run_index", "seed", "score"] + metric_keys
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()
    for row in runs:
        out = {
            "run_index": row["run_index"],
            "seed": row["seed"],
            "score": row["score"],
        }
        out.update(row["metrics"])
        writer.writerow(out)
PY

safe_cleanup_dir "${EVAL_TMP_DIR}"

echo "Finished ${NUM_RUNS} train+eval runs." | tee -a "${LOG_FILE}"
echo "Summary JSON: ${SUMMARY_JSON}" | tee -a "${LOG_FILE}"
echo "Summary CSV:  ${SUMMARY_CSV}" | tee -a "${LOG_FILE}"
echo "Final model:  ${MODEL_DIR}" | tee -a "${LOG_FILE}"
