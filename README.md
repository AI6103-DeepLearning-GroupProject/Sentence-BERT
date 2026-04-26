# Beyond SBERT 0.2.4

This repository is an experimental fork of the original
[`sentence-transformers`](https://github.com/huggingface/sentence-transformers) 0.2.4 codebase. It keeps the old SBERT training stack on 2019-era.
(`transformers==2.2.1`, PyTorch, custom readers, and the original model
serialization format), then adds a config-driven pipeline for reproducible
STSBenchmark-focused experiments.

The main target is **STSBenchmark test-set Spearman correlation** with staged
training over NLI, contrastive MNRL, and STS objectives.

## What Changed From SBERT 0.2.4

- **Config-driven training**: `examples/train_with_config.py` reads
  `examples/config.yaml` and supports both legacy experiments and staged runtime
  experiments.
- **Runtime experiment modes**: `sts`, `nli`, `nli_mnrl`, `sts_nli`,
  `sts_nli_mnrl`, `sts_nli_mnrl_cosent`, `sts_nli_mnrl_aoe`,
  `wikipedia_triplet`, and `wikipedia_triplet_mnrl`.
- **Reproducibility controls**: explicit seed setting, deterministic cuDNN
  toggles, centralized optimizer settings, fixed stage order, and multi-run
  result aggregation.
- **Improved MNRL**: `MultipleNegativesRankingLoss` now uses normalized cosine
  scores with `scale=20.0`, matching the later sentence-transformers behavior
  more closely than the original raw dot-product implementation.
- **NLI hard negatives for MNRL**: `examples/datasets/build_contrastive_pairs.py`
  builds triplets from AllNLI: premise as anchor, entailment as positive, and
  contradiction as explicit hard negative.
- **CoSENT STS objective**: `CoSENTLoss` is available as a stronger STS ranking
  loss and is enabled only by `sts_nli_mnrl_cosent`.
- **AoE experimental loss**: `AoECombinedLoss` is available for the
  `sts_nli_mnrl_aoe` mode.
- **Evaluation scripts**: STS, triplet, and SentEval evaluators write JSON/CSV
  summaries under `result/`.

Recent repository history reflects these additions, including MNRL hard-negative
work, CoSENT experiments, AoE experiments, SentEval support, and Slurm
reproduction scripts.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `config.yaml` | Top-level runtime configuration for Slurm train/eval jobs. |
| `examples/config.yaml` | Full experiment definitions, model variants, data modes, and stage settings. |
| `examples/train_with_config.py` | Main config-driven training entry point. |
| `examples/evaluate_pretrained.py` | STSBenchmark evaluator for a local or remote SentenceTransformer model. |
| `examples/evaluate_triplet_pretrained.py` | Triplet evaluator. |
| `examples/evaluate_senteval.py` | SentEval transfer-task evaluator. |
| `examples/datasets/get_data.py` | Downloads AllNLI, STSBenchmark, and Wikipedia triplets. |
| `examples/datasets/build_contrastive_pairs.py` | Builds MNRL pairs or NLI hard-negative triplets. |
| `sentence_transformers/losses/` | Softmax, cosine, MNRL, CoSENT, AoE, and triplet losses. |
| `scripts/02_train_sbert.slurm` | Main train+eval Slurm job. |
| `scripts/03_sts_eval.slurm` | Standalone STS evaluation Slurm job. |
| `result/` | Reported JSON summaries and generated run artifacts. |

## Environment

The code is based on the old SBERT 0.2.4 stack. Use the pinned dependency set
instead of a modern sentence-transformers install.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Dependencies are defined in `requirements.txt`:

```text
transformers==2.2.1
torch>=1.0.1
tqdm
numpy
scikit-learn
scipy
nltk
PyYAML
```

For cluster runs, the provided Slurm scripts assume a virtual environment at
`$HOME/venvs/sbert024` unless `VENV_DIR` is overridden.

## Data Setup

Download the datasets:

```bash
python examples/datasets/get_data.py
```

This creates datasets under `examples/datasets/`. The Slurm scripts also try to
sync or normalize dataset locations into:

```text
datasets/AllNLI/
datasets/stsbenchmark/
datasets/wikipedia-sections-triplets/
```

Expected STS files:

```text
datasets/stsbenchmark/sts-train.csv
datasets/stsbenchmark/sts-dev.csv
datasets/stsbenchmark/sts-test.csv
```

Expected AllNLI files:

```text
datasets/AllNLI/s1.train.gz
datasets/AllNLI/s2.train.gz
datasets/AllNLI/labels.train.gz
```

For MNRL modes, the training script can auto-build:

```text
datasets/contrastive/pairs_train.tsv
datasets/contrastive/nli_pairs_train.tsv
```

Hard-negative MNRL files should have **three tab-separated columns**:

```text
anchor    positive_entailment    hard_negative_contradiction
```

Check the format with:

```bash
awk -F '\t' 'NR==1{print NF; exit}' datasets/contrastive/pairs_train.tsv
```

Expected output for hard-negative MNRL modes is `3`.

## Configuration

### Top-Level Runtime Config

`config.yaml` controls Slurm-oriented runs:

```yaml
slurm:
  train:
    config_path: examples/config.yaml
    experiment: runtime
    model_variant: sbert_base
    data_mode: sts_nli_mnrl_cosent
    epochs: 4
    seed: 42
    num_runs: 1
    seed_step: 1
    resume_stage: 1
    result_dir: ""
  eval:
    task_type: sts
    model_name: result/runtime_sbert_base_sts_nli_mnrl_cosent/model
    sts_path: datasets/stsbenchmark
    sts_test_file: sts-test.csv
    batch_size: 64
    main_similarity: cosine
```

> Why 42? Go ask Douglas Adams. What we know is that this numbers answers the Ultimate Question of Universe, just like AI networks we're training.

### Model Variants

Runtime model variants are defined in `examples/config.yaml`:

| Key | Encoder | Base Model | Pooling |
| --- | --- | --- | --- |
| `sbert_base` | BERT | `bert-base-uncased` | mean pooling |
| `sbert_large` | BERT | `bert-large-uncased` | mean pooling |
| `sroberta_base` | RoBERTa | `roberta-base` | mean pooling |
| `sroberta_large` | RoBERTa | `roberta-large` | mean pooling |

> We focused on SBERT-base mainly

### Runtime Data Modes

| Mode | Stages | Main Use |
| --- | --- | --- |
| `sts` | STS cosine | STS-only baseline. |
| `nli` | NLI softmax | NLI sentence embedding baseline. |
| `nli_mnrl` | NLI softmax → hard-negative MNRL | Contrastive NLI-only variant. |
| `sts_nli` | NLI softmax → STS cosine | Original SBERT-style NLI then STS. |
| `sts_nli_mnrl` | NLI softmax → hard-negative MNRL → STS cosine | Main MNRL-enhanced STS mode. |
| `sts_nli_mnrl_cosent` | NLI softmax → hard-negative MNRL → STS CoSENT | Main CoSENT STS ranking mode. |
| `sts_nli_mnrl_aoe` | NLI softmax → MNRL/AoE joint → STS cosine | Experimental AoE mode. |
| `wikipedia_triplet` | Wikipedia triplet | Topic triplet baseline. |
| `wikipedia_triplet_mnrl` | Wikipedia MNRL → Wikipedia triplet | Wikipedia contrastive variant. |

> Only `sts_nli_mnrl_cosent` replaces the final STS loss with CoSENT. Other modes
keep their existing losses.

### Attention Pooling

Pooling varients are also defined in `examples/config.yaml`, you can switch pooling type by changing the config.

## Reproducible Local Runs

### Train the Current Default Mode

> If you get access to TC2 HPC, simply submit the 01 and 02 Slurm scripts with runtime config at `config.yaml`. Output path for model, summary and logs can be found under results/ in the server. For local runs, use the command below.

```bash
python examples/train_with_config.py \
  --config examples/config.yaml \
  --experiment runtime \
  --model-variant sbert_base \
  --data-mode sts_nli_mnrl \
  --seed 42 \
  --epochs 4 \
  --output-path result/runtime_sbert_base_sts_nli_mnrl_cosent/model
```

Important details:

- Runtime stage-specific `num_epochs` overrides `--epochs`.
- For `sts_nli_mnrl_cosent`, the actual stage epochs are `1` NLI, `2` MNRL,
  and `1` STS CoSENT.
- `save_best_model=true` saves the best model according to the dev evaluator in
  the final stage.
- If `use_hard_negative: true` is configured but the contrastive TSV has only
  two columns, training fails instead of silently falling back to normal MNRL.

### Evaluate STSBenchmark

```bash
python examples/evaluate_pretrained.py \
  --model-name result/runtime_sbert_base_sts_nli_mnrl_cosent/model \
  --sts-path datasets/stsbenchmark \
  --sts-test-file sts-test.csv \
  --batch-size 64 \
  --seed 42 \
  --num-runs 1 \
  --main-similarity cosine \
  --output-path result/eval_runtime_sbert_base_sts_nli_mnrl_cosent
```

The evaluator writes:

```text
result/eval_runtime_sbert_base_sts_nli_mnrl_cosent/summary.json
result/eval_runtime_sbert_base_sts_nli_mnrl_cosent/summary_table.csv
```

The primary STS metric is cosine Spearman correlation.

## Reproducible Slurm Runs

### Main Train + Eval Job

Edit `config.yaml`, then submit:

```bash
sbatch scripts/02_train_sbert.slurm
```

The job reads:

- `slurm.train.*` for training.
- `slurm.eval.*` for evaluation.

By default, artifacts are written to:

```text
result/runtime_<model_variant>_<data_mode>/
result/runtime_<model_variant>_<data_mode>/model/
result/runtime_<model_variant>_<data_mode>/summary.json
result/runtime_<model_variant>_<data_mode>/summary_table.csv
result/logs/
```

For the current default mode:

```text
result/runtime_sbert_base_sts_nli_mnrl_cosent/
```

### Standalone STS Evaluation

```bash
sbatch scripts/03_sts_eval.slurm
```

Override a model path if needed:

```bash
MODEL_NAME=result/runtime_sbert_base_sts_nli_mnrl_cosent/model \
sbatch scripts/03_sts_eval.slurm
```

### Multi-Run Reproduction

Set `num_runs` and `seed_step` in `config.yaml`:

```yaml
slurm:
  train:
    seed: 42
    num_runs: 3
    seed_step: 1
```

This produces runs with seeds `42`, `43`, and `44`, then aggregates the mean and
standard deviation in `summary.json`.

## Reported Result Files

Existing JSON summaries are stored in `result/`. They are treated as run outputs,
not as source-of-truth configuration. Example files include:

| File | Task |
| --- | --- |
| `result/sbert-base-sts.json` | STS baseline. |
| `result/sbert-base-sts-nli.json` | NLI + STS. |
| `result/sbert-base-sts-nli-mnrl.json` | NLI + hard-negative MNRL + STS. |
| `result/sbert-base-nli-mnrl-cosent.json` | NLI + hard-negative MNRL + STS CoSENT. |
| `result/sbert-base-sts-nli-mnrl-aoe.json` | AoE experimental mode. |
| `result/sbert-base-nli-senteval.json` | SentEval NLI baseline. |
| `result/sbert-base-nli-mnrl-senteval.json` | SentEval NLI + MNRL. |

When comparing results, verify:

- Same `data_mode`.
- Same model variant.
- Same seed list.
- Same STS test file.
- Same `main_similarity`.
- Whether the MNRL TSV has two columns or three columns.

## Implementation Notes

### Staged Training

`examples/train_with_config.py` resolves runtime plans from
`examples/config.yaml`. Each stage creates its own dataloader, loss, evaluator,
and training configuration. The same model object is carried across stages.

### MNRL With Hard Negatives

`MultipleNegativesRankingLoss` accepts either:

- two texts per example: `anchor, positive`; or
- three or more texts per example: `anchor, positive, hard_negative...`.

The loss normalizes embeddings, computes scaled cosine scores, appends explicit
hard negatives to the in-batch candidate matrix, and uses cross entropy where the
positive target remains the diagonal item in the first candidate block.

### CoSENT

`CoSENTLoss` is a pairwise ranking loss over STS labels. It compares score
differences between sentence pairs inside a batch and is used by the
`sts_cosent` task type. The current scale is `20.0`.

### Determinism

`sentence_transformers/util.py` sets:

- Python random seed.
- NumPy random seed.
- PyTorch CPU and CUDA seeds.
- cuDNN deterministic and benchmark flags.

The default seed is `42`.

## Common Failure Modes

- **Old two-column MNRL cache**: delete `datasets/contrastive/pairs_train.tsv`
  and rerun the Slurm job, or ensure the first row has three columns.
- **Wrong mode evaluated**: check `config.yaml` `slurm.eval.model_name`; it must
  match the trained `slurm.train.data_mode`.
- **Resume skipped a new stage**: set `resume_stage: 1` when comparing methods
  from scratch.
- **Missing local model path**: evaluation treats `result/.../model` as a local
  directory and fails if it does not exist.
- **Dataset path mismatch**: make sure STS files exist under
  `datasets/stsbenchmark/`.

## Citation

This fork builds on Sentence-BERT:

```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "http://arxiv.org/abs/1908.10084",
}
```
