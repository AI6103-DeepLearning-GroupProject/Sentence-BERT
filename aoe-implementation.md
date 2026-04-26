# AoE Implementation

## Goal

Add a clean AoE-lite path so the repo can run direct AoE experiments without relying on the existing `sts_nli_mnrl_aoe` path.

Target comparisons:

1. `sbert_base + sts_aoe`
2. `sbert_base + sts_nli_aoe`

## Why AoE-lite First

The repo already contains an AoE-flavored stage in `sts_nli_mnrl_aoe`, but that path is not the clean baseline we want.

Current limitations:

- it is stacked on top of `nli -> mnrl`
- it uses `AoECombinedLoss`, which is `angle + contrastive`
- it depends on `aoe_pairs_train.tsv`
- the referenced `examples/datasets/build_aoe_pairs.py` helper is missing from the repo

AoE-lite avoids those issues by reusing the existing STS dataset directly.

## Scope

AoE-lite in this repo means:

- `L_cos`: CoSENT-style ranking loss over cosine similarity scores
- `L_angle`: CoSENT-style ranking loss over angle-derived similarity scores

Total loss:

```text
L = cosine_weight * L_cos + angle_weight * L_angle
```

This first pass intentionally does not add an in-batch-negative term.

## Files Added Or Updated

- `sentence_transformers/losses/AoELiteLoss.py`
- `sentence_transformers/losses/__init__.py`
- `examples/train_with_config.py`
- `examples/config.yaml`
- `scripts/02_train_sbert.slurm`
- `config.yaml`

## New Runtime Modes

### `sts_aoe`

One-stage STS training with `AoELiteLoss`.

### `sts_nli_aoe`

Two-stage training:

1. `nli_softmax`
2. `sts_aoe`

This mirrors the existing `sts_nli` schedule but swaps the STS stage loss.

## Default AoE-lite Hyperparameters

- `cosine_weight: 1.0`
- `angle_weight: 0.02`
- `cosine_tau: 20.0`
- `angle_tau: 20.0`

These are intentionally conservative so the new angle term does not dominate the existing STS signal on the first pass.

## Implementation Notes

### Cosine ranking term

The cosine term uses the STS float labels directly as ranking targets.
If pair `j` has a higher similarity label than pair `i`, the model is trained so pair `j` gets a higher cosine score.

### Angle ranking term

The angle term splits the sentence embedding into real and imaginary halves.
It computes an angle-difference score in complex space and ranks pairs so more similar examples receive smaller angle differences.

### Even embedding requirement

AoE-lite requires the sentence embedding dimension to be even.
The current pooling dimensions in this repo satisfy that requirement.

## How To Run

Example direct STS AoE run:

```bash
python examples/train_with_config.py --experiment runtime --model-variant sbert_base --data-mode sts_aoe
```

Example staged NLI then AoE run:

```bash
python examples/train_with_config.py --experiment runtime --model-variant sbert_base --data-mode sts_nli_aoe
```

SLURM mode names are the same:

- `sts_aoe`
- `sts_nli_aoe`

## Recommended Comparison Matrix

Compare these under the same STS evaluation pipeline:

1. `sbert_base + sts`
2. `sbert_base + sts_nli`
3. `sbert_base + sts_nli_mnrl`
4. `sbert_base + sts_aoe`
5. `sbert_base + sts_nli_aoe`

Use 3 seeds if you want report-quality numbers:

- `42`
- `43`
- `44`

## Known Risks

- the angle term can be sensitive to scale and temperature
- ranking loss can become noisy when many labels in a batch are nearly tied
- mixing AoE with MNRL is still a separate question and should be evaluated after the direct AoE baselines

## Deliberate Non-Goals In This Pass

- no change to the existing `AoECombinedLoss`
- no change to `sts_nli_mnrl_aoe`
- no new AoE pair-building script
- no report-table updates yet
