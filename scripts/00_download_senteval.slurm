#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#
### Specify Memory allocate to this job ###
#SBATCH --mem=24G
#
### Specify number of core (CPU) to allocate to per task ###
#SBATCH --cpus-per-task=2
#
### Specify number of node to compute ###
#SBATCH --nodes=1
#
### Optional: Specify node to execute the job ###
### Remove 1st # at next line for the option to take effect ###
#SBATCH --nodelist=TC2N06
#
### Specify Time Limit ###
#SBATCH --time=120
#
#SBATCH --job-name=senteval_setup
#SBATCH --output=result/logs/%x-%j.out
#SBATCH --error=result/logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PYTHON_MODULE="${PYTHON_MODULE:-python312}"
SENTEVAL_DIR="${SENTEVAL_DIR:-datasets/SentEval}"
SENTEVAL_DATA_DIR="${SENTEVAL_DATA_DIR:-${SENTEVAL_DIR}/data/downstream}"

module purge
module load "${PYTHON_MODULE}"

cd "${PROJECT_DIR}"
mkdir -p result/logs datasets

if ! command -v git >/dev/null 2>&1; then
  echo "git not found in PATH; required for cloning SentEval." >&2
  exit 1
fi

if [ ! -d "${SENTEVAL_DIR}/.git" ]; then
  echo "Cloning SentEval into ${SENTEVAL_DIR} ..."
  git clone --depth 1 https://github.com/facebookresearch/SentEval.git "${SENTEVAL_DIR}"
else
  echo "SentEval repository already exists at ${SENTEVAL_DIR}. Skip clone."
fi

python - "${SENTEVAL_DATA_DIR}" <<'PY'
import io
import os
import shutil
import ssl
import sys
import tarfile
import time
import urllib.request
import zipfile


def _download(url: str, output_path: str, retries: int = 3) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ctx = ssl.create_default_context()
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            print(f"[download] {url}")
            with urllib.request.urlopen(url, context=ctx, timeout=120) as r:
                content = r.read()
            with open(output_path, "wb") as f_out:
                f_out.write(content)
            return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"[warn] attempt {attempt}/{retries} failed: {exc}")
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download {url}") from last_err


def _extract_zip(zip_path: str, dst_dir: str) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst_dir)


def _extract_tar_gz(tar_path: str, dst_dir: str) -> None:
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(dst_dir)


def _copy_text(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src, "r", encoding="latin-1", errors="ignore") as f_in, open(dst, "w", encoding="utf-8") as f_out:
        for line in f_in:
            f_out.write(line.rstrip("\n\r") + "\n")


def _find_extracted_file(root_dir: str, candidate_rel_paths, required_basename: str) -> str:
    for rel_path in candidate_rel_paths:
        candidate = os.path.join(root_dir, rel_path)
        if os.path.isfile(candidate):
            return candidate

    matches = []
    for walk_root, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == required_basename:
                matches.append(os.path.join(walk_root, filename))

    if not matches:
        raise FileNotFoundError(f"Cannot find extracted file '{required_basename}' under {root_dir}")

    # Prefer paths that contain the expected SentEval data folders.
    preferred_tokens = ("data_bin_classif", "/data/", "\\data\\")
    for match in matches:
        normalized = match.replace("\\", "/")
        if any(token in normalized for token in preferred_tokens):
            return match

    return matches[0]


def _download_simple_file(url: str, dst: str) -> None:
    if os.path.isfile(dst):
        print(f"[skip] {dst} exists")
        return
    tmp_path = dst + ".tmp"
    _download(url, tmp_path)
    os.replace(tmp_path, dst)


def ensure_mr_cr_subj_mpqa(data_path: str) -> None:
    expected = [
        os.path.join(data_path, "MR", "rt-polarity.pos"),
        os.path.join(data_path, "MR", "rt-polarity.neg"),
        os.path.join(data_path, "CR", "custrev.pos"),
        os.path.join(data_path, "CR", "custrev.neg"),
        os.path.join(data_path, "SUBJ", "subj.subjective"),
        os.path.join(data_path, "SUBJ", "subj.objective"),
        os.path.join(data_path, "MPQA", "mpqa.pos"),
        os.path.join(data_path, "MPQA", "mpqa.neg"),
    ]
    if all(os.path.isfile(p) for p in expected):
        print("[skip] MR/CR/SUBJ/MPQA already prepared")
        return

    work_dir = os.path.join(data_path, "_tmp_binclassif")
    os.makedirs(work_dir, exist_ok=True)
    zip_path = os.path.join(work_dir, "datasmall_NB_ACL12.zip")
    _download("https://dl.fbaipublicfiles.com/senteval/senteval_data/datasmall_NB_ACL12.zip", zip_path)
    _extract_zip(zip_path, work_dir)

    mapping = [
        (["data_bin_classif/data/rt10662/rt-polarity.pos"], "rt-polarity.pos", "MR/rt-polarity.pos"),
        (["data_bin_classif/data/rt10662/rt-polarity.neg"], "rt-polarity.neg", "MR/rt-polarity.neg"),
        (["data_bin_classif/data/customerr/custrev.pos"], "custrev.pos", "CR/custrev.pos"),
        (["data_bin_classif/data/customerr/custrev.neg"], "custrev.neg", "CR/custrev.neg"),
        (["data_bin_classif/data/subj/subj.subjective"], "subj.subjective", "SUBJ/subj.subjective"),
        (["data_bin_classif/data/subj/subj.objective"], "subj.objective", "SUBJ/subj.objective"),
        (["data_bin_classif/data/mpqa/mpqa.pos"], "mpqa.pos", "MPQA/mpqa.pos"),
        (["data_bin_classif/data/mpqa/mpqa.neg"], "mpqa.neg", "MPQA/mpqa.neg"),
    ]

    for rel_candidates, required_basename, rel_dst in mapping:
        src = _find_extracted_file(work_dir, rel_candidates, required_basename)
        dst = os.path.join(data_path, rel_dst)
        _copy_text(src, dst)

    shutil.rmtree(work_dir, ignore_errors=True)
    print("[ok] MR/CR/SUBJ/MPQA prepared")


def ensure_sst2(data_path: str) -> None:
    out_dir = os.path.join(data_path, "SST", "binary")
    os.makedirs(out_dir, exist_ok=True)
    for split in ("train", "dev", "test"):
        dst = os.path.join(out_dir, f"sentiment-{split}")
        url = f"https://raw.githubusercontent.com/PrincetonML/SIF/master/data/sentiment-{split}"
        _download_simple_file(url, dst)
    print("[ok] SST2 prepared")


def ensure_trec(data_path: str) -> None:
    out_dir = os.path.join(data_path, "TREC")
    os.makedirs(out_dir, exist_ok=True)
    files = {
        "train_5500.label": "http://cogcomp.cs.illinois.edu/Data/QA/QC/train_5500.label",
        "TREC_10.label": "http://cogcomp.cs.illinois.edu/Data/QA/QC/TREC_10.label",
    }
    for name, url in files.items():
        dst = os.path.join(out_dir, name)
        _download_simple_file(url, dst)
        # Match original script cleanup.
        with open(dst, "r", encoding="latin-1", errors="ignore") as f_in:
            text = f_in.read().replace("`", "").replace("'", "")
        with open(dst, "w", encoding="utf-8") as f_out:
            f_out.write(text)
    print("[ok] TREC prepared")


def ensure_mrpc(data_path: str) -> None:
    out_dir = os.path.join(data_path, "MRPC")
    os.makedirs(out_dir, exist_ok=True)
    files = {
        "msr_paraphrase_train.txt": "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt",
        "msr_paraphrase_test.txt": "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt",
    }
    for name, url in files.items():
        _download_simple_file(url, os.path.join(out_dir, name))
    print("[ok] MRPC prepared")


def main() -> None:
    data_path = os.path.abspath(sys.argv[1])
    os.makedirs(data_path, exist_ok=True)
    print(f"[info] SentEval downstream path: {data_path}")

    ensure_mr_cr_subj_mpqa(data_path)
    ensure_sst2(data_path)
    ensure_trec(data_path)
    ensure_mrpc(data_path)

    required = [
        "MR/rt-polarity.pos",
        "MR/rt-polarity.neg",
        "CR/custrev.pos",
        "CR/custrev.neg",
        "SUBJ/subj.subjective",
        "SUBJ/subj.objective",
        "MPQA/mpqa.pos",
        "MPQA/mpqa.neg",
        "SST/binary/sentiment-train",
        "SST/binary/sentiment-dev",
        "SST/binary/sentiment-test",
        "TREC/train_5500.label",
        "TREC/TREC_10.label",
        "MRPC/msr_paraphrase_train.txt",
        "MRPC/msr_paraphrase_test.txt",
    ]

    missing = [p for p in required if not os.path.isfile(os.path.join(data_path, p))]
    if missing:
        raise RuntimeError("Missing SentEval files:\n  - " + "\n  - ".join(missing))

    print("[done] SentEval toolkit + selected downstream datasets are ready.")


if __name__ == "__main__":
    main()
PY

echo "SentEval setup complete:"
echo "  Toolkit: ${SENTEVAL_DIR}"
echo "  Data:    ${SENTEVAL_DATA_DIR}"
