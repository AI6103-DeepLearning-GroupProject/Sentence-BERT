# AI6103 实验结果总结（中文）

日期：2026-04-11  
仓库：Sentence-BERT  
目的：用于小组交流讨论，快速对齐三条改进路线的收益、风险与后续行动。

## 1. 实验范围与设置

本轮完成了三条路线的单次 seed=42 运行：

1. 方向一（Attention Pooling，监督 STS）
- 训练脚本：examples/training_stsbenchmark_bert.py
- 关键参数：--pooling attention
- 结果目录：output/training_stsbenchmark_bert-attention-seed42-2026-04-11_08-00-49

2. 方向二（SimCSE 风格无监督对比学习）
- 训练脚本：examples/training_simcse_unsupervised.py
- 关键参数：pair_mode=dropout_and_aug, augmentation=random_delete_swap
- 结果目录：output/training_simcse_unsup-dropout_and_aug-random_delete_swap-seed42-2026-04-11_08-26-03

3. 方向三（Task-aware Prompt，多任务）
- 训练脚本：examples/training_task_aware_prompt.py
- 关键参数：--use_prompt --tasks sts,nli,unsup
- 结果目录：output/training_task_aware_prompt-prompt-nli-sts-unsup-seed42-2026-04-11_08-26-58

## 2. 核心结果（用于组会快速对比）

| 方向 | 主要目标 | 关键指标 | 本轮结果 |
| --- | --- | --- | --- |
| Direction-1 Attention Pooling | STS 监督优化 | STS dev cosine Spearman | **0.8750** |
| Direction-2 SimCSE 无监督 | 无标注语义表示 | STS dev cosine Spearman | **0.7966** |
| Direction-3 Prompt 多任务 | STS + NLI 平衡 | STS dev / STS test / NLI dev acc | **0.8609 / 0.8297 / 0.6334** |

## 3. 结论先行

1. 如果目标是“单任务 STS 指标最大化”，方向一当前最优（0.8750）。
2. 如果目标是“无标注可迁移语义基线”，方向二可作为低成本起点，但本轮上限明显低于监督路线。
3. 如果目标是“多任务可用性（STS + NLI）”，方向三最均衡，尤其在需要兼顾语义相似度与推理能力时更实用。

## 4. 结果解读（讨论重点）

1. 方向一为何强：
- Attention Pooling 能学习 token 级别重要性，在 STS 这种细粒度语义匹配任务上收益直观。

2. 方向二为何偏弱：
- 无监督对比学习对数据分布与增广策略较敏感；当前仅单次 seed、单组增强，可能尚未达到最佳组合。

3. 方向三为何值得保留：
- 通过任务前缀把编码目标显式化，能在同一编码器里保持多任务能力；虽然 STS 单点不及方向一，但综合指标与落地价值更高。

## 5. 风险与局限

1. 当前结果均为单 seed（42），统计稳定性不足。
2. 三条路线训练预算并非完全等价（epoch/目标函数/任务数差异），横向比较需谨慎。
3. 方向二仅报告 dev 结果，建议补充统一 test 评测以便与方向一/三更公平比较。

## 6. 建议的下一步（小组可直接分工）

1. 稳定性验证：三条路线补 3-seed（42/43/44），报告均值与标准差。
2. 公平性验证：统一训练步数或 wall-clock 预算后再做一次主对比。
3. 方向二增强：网格搜索增广强度（delete_ratio, swap_ratio）与 pair_mode。
4. 方向三增强：加入 prompt 消融（去掉任务前缀、共享前缀、任务专属前缀）验证真实收益来源。
5. 汇报落地：准备“单任务最优（方向一）+ 多任务最优（方向三）”双模型交付方案。

## 7. 组会可直接引用的一句话

- STS 单任务最优：Direction-1（0.8750）。
- 多任务平衡最优：Direction-3（STS test 0.8297, NLI dev acc 0.6334）。
- 无监督可用基线：Direction-2（0.7966），后续仍有优化空间。
