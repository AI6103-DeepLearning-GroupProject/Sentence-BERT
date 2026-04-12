# AI6103 Experiment Summary (English)

Date: 2026-04-11  
Repository: Sentence-BERT  
Purpose: team communication and discussion on gains, trade-offs, and next actions across the three directions.

## 1. Scope and Run Setup

We completed one run (seed=42) for each direction:

1. Direction-1 (Attention Pooling, supervised STS)
- Training script: examples/training_stsbenchmark_bert.py
- Key option: --pooling attention
- Output: output/training_stsbenchmark_bert-attention-seed42-2026-04-11_08-00-49

2. Direction-2 (SimCSE-style unsupervised contrastive learning)
- Training script: examples/training_simcse_unsupervised.py
- Key setup: pair_mode=dropout_and_aug, augmentation=random_delete_swap
- Output: output/training_simcse_unsup-dropout_and_aug-random_delete_swap-seed42-2026-04-11_08-26-03

3. Direction-3 (Task-aware Prompt, multi-task)
- Training script: examples/training_task_aware_prompt.py
- Key option: --use_prompt --tasks sts,nli,unsup
- Output: output/training_task_aware_prompt-prompt-nli-sts-unsup-seed42-2026-04-11_08-26-58

## 2. Key Results (Quick Comparison)

| Direction | Main objective | Key metric | Result |
| --- | --- | --- | --- |
| Direction-1 Attention Pooling | Maximize supervised STS | STS dev cosine Spearman | **0.8750** |
| Direction-2 SimCSE unsupervised | Label-free semantic representation | STS dev cosine Spearman | **0.7966** |
| Direction-3 Prompt multi-task | Balance STS + NLI | STS dev / STS test / NLI dev acc | **0.8609 / 0.8297 / 0.6334** |

## 3. Executive Takeaways

1. If the priority is best single-task STS performance, Direction-1 is currently the top choice (0.8750).
2. If the priority is label-free training and quick semantic baseline, Direction-2 is useful but clearly below supervised alternatives in this run.
3. If the priority is balanced multi-task usability, Direction-3 is the strongest practical option.

## 4. Interpretation for Team Discussion

1. Why Direction-1 is strong:
- Attention pooling learns token importance and helps fine-grained semantic matching in STS.

2. Why Direction-2 is weaker now:
- Unsupervised contrastive learning is sensitive to data quality and augmentation policy; we only tested one seed and one augmentation setting.

3. Why Direction-3 matters:
- Task prefixes explicitly guide encoding behavior and preserve broader capability (STS + NLI), even if pure STS peak is slightly below Direction-1.

## 5. Risks and Limitations

1. All results are from single-seed runs (seed=42), so variance is unknown.
2. Training budgets are not fully matched (different objectives/tasks/epochs), which affects strict fairness.
3. Direction-2 currently reports dev-side performance only; unified test evaluation should be added.

## 6. Recommended Next Steps

1. Stability check: run 3 seeds (42/43/44) for all directions and report mean ± std.
2. Fairness check: align training budget (steps or wall-clock time) and rerun comparison.
3. Direction-2 improvement: grid search augmentation strengths and pair modes.
4. Direction-3 ablation: compare no-prompt vs shared prompt vs task-specific prompt.
5. Delivery strategy: keep two candidates for presentation/deployment:
- Direction-1 as best STS specialist.
- Direction-3 as best balanced multi-task model.

## 7. One-line Summary for Meeting Slides

- Best STS specialist: Direction-1 (0.8750 dev Spearman).
- Best balanced model: Direction-3 (0.8297 STS test, 0.6334 NLI dev acc).
- Practical unsupervised baseline: Direction-2 (0.7966 dev), with clear room for tuning.
