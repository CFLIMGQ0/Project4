# TASK1 实验结果汇总

整理时间：2026-05-21

## 检查范围与结论

本次只检查 TASK1 胃镜三标签 examination-level multi-label classification 相关实验，不纳入 TASK2 结果。当前可用于论文实验部分的正式结果集中在：

```text
/home/Lim/Project4/outputs/train_runs/task1/task1_ablation/exp_task1_auto_module_instance_search/
```

该目录包含三组完整实验：标签图推理模块消融、输入实例数消融、最终 11 个模型对比。最终模型对比中，`Label graph MIL` 在测试集 macro F1、micro F1 和 kappa 上取得最优结果；其中 macro F1 为 `0.8684`，相对最佳非本文模型 `Top-k MIL` 的 `0.8606` 提升约 `0.0078`。需要注意的是，macro AUROC 和 macro AUPRC 的最高值分别来自 `Transformer-context MIL` 和 `DSMIL`，因此论文中更适合将 macro F1 / micro F1 作为主结果，将 AUROC / AUPRC 作为补充指标。

另外检查到两个历史结果目录：

| 目录 | 状态 | 建议 |
|---|---|---|
| `/home/Lim/Project4/outputs/train_runs/task1/task1_ablation/exp_task1_ablation1/` | 只完成 5 个模块消融，汇总文件仍有 TODO | 不作为论文主表 |
| `/home/Lim/Project4/outputs/train_runs/gastro_multilabel_task/` | 早期 TASK1 结果，目录结构和当前 TASK1 规范不完全一致 | 可作为排查参考，不建议混入正式结果 |

## 实验设置摘要

| 项目 | 设置 |
|---|---|
| 任务 | TASK1 胃镜三标签检查级多标签分类 |
| 标签 | 食管 SMT、食管黏膜/肿物病变、胃炎 |
| 样本数 | 2863 次检查 |
| 图像数 | 171642 张 |
| 数据划分 | train/val/test = 0.6/0.2/0.2 |
| 实际划分 | 1717 / 572 / 574 |
| 主干网络 | ConvNeXt-Tiny |
| 图像尺寸 | 224 |
| 最大训练轮数 | 30 |
| 优化器 | AdamW |
| 初始学习率 | 2e-5 |
| 损失函数 | Asymmetric loss |
| 训练采样 | random instance sampling |
| 验证/测试采样 | uniform instance sampling |
| 结果选择 | `best_macro_f1` checkpoint |
| 阈值 | 0.5 |

## 数据划分与标签分布

| 标签 | 训练集阳性 | 训练集阴性 | 验证集阳性 | 验证集阴性 | 测试集阳性 | 测试集阴性 |
|---|---:|---:|---:|---:|---:|---:|
| 食管 SMT | 836 | 881 | 263 | 309 | 276 | 298 |
| 食管黏膜/肿物病变 | 883 | 834 | 306 | 266 | 300 | 274 |
| 胃炎 | 701 | 1016 | 241 | 331 | 257 | 317 |

## 表 1. 最终模型对比

| Rank | Method | Type | Macro F1 | Micro F1 | Macro AUROC | Macro AUPRC | Subset Acc | Kappa | Best epoch |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Label graph MIL | Ours | **0.8684** | **0.8655** | 0.9187 | 0.9099 | 0.7056 | **0.7276** | 24 |
| 2 | Top-k MIL | Baseline | 0.8606 | 0.8565 | 0.9220 | 0.9106 | 0.6620 | 0.7037 | 11 |
| 3 | TransMIL | SOTA | 0.8606 | 0.8578 | 0.9247 | 0.9179 | **0.7143** | 0.7147 | 16 |
| 4 | DSMIL | SOTA | 0.8601 | 0.8570 | 0.9241 | **0.9181** | 0.6951 | 0.7102 | 27 |
| 5 | Transformer-context MIL | Baseline | 0.8589 | 0.8555 | **0.9253** | 0.9181 | 0.6812 | 0.7057 | 11 |
| 6 | Mean pooling | Baseline | 0.8584 | 0.8555 | 0.9190 | 0.9102 | 0.6742 | 0.7035 | 24 |
| 7 | Attention MIL | Baseline | 0.8504 | 0.8469 | 0.9179 | 0.9164 | 0.6672 | 0.6894 | 22 |
| 8 | DTFD-MIL | SOTA | 0.8452 | 0.8405 | 0.9177 | 0.9098 | 0.5714 | 0.6583 | 25 |
| 9 | CLAM-MB | SOTA | 0.8360 | 0.8321 | 0.9071 | 0.9000 | 0.5958 | 0.6496 | 22 |
| 10 | CLAM-SB | SOTA | 0.8284 | 0.8236 | 0.9070 | 0.8907 | 0.5453 | 0.6203 | 15 |
| 11 | Max pooling | Baseline | 0.7533 | 0.7401 | 0.8439 | 0.8416 | 0.1533 | 0.3544 | 14 |

## 表 2. 标签图推理模块消融

该组实验固定 `train_max_instances=16`，保持 ConvNeXt-Tiny、标签级注意力聚合、分类头、训练策略和数据划分一致，仅替换标签关系建模模块。

| Module | Graph type | train max instances | Macro F1 | Micro F1 | Best epoch |
|---|---|---:|---:|---:|---:|
| Full Label Graph Reasoner | learnable | 16 | 0.8598 | 0.8548 | 13 |
| w/o Label Graph Reasoner | none | 16 | 0.8539 | 0.8483 | 11 |
| Label Self-Attention Reasoner | self_attention | 16 | 0.8560 | 0.8525 | 25 |
| Static Co-occurrence GCN Reasoner | static_gcn | 16 | 0.8629 | 0.8604 | 17 |
| Dynamic Label GAT Reasoner | dynamic_gat | 16 | 0.8636 | 0.8597 | 13 |
| Label Transformer Reasoner | label_transformer | 16 | 0.8547 | 0.8511 | 16 |
| Low-Rank Label Graph Reasoner | low_rank_graph | 16 | 0.8506 | 0.8484 | 24 |
| Cosine Dynamic Graph Reasoner | cosine_graph | 16 | 0.8636 | 0.8610 | 17 |
| Label MLP-Mixer Reasoner | label_mlp_mixer | 16 | 0.8620 | 0.8588 | 23 |
| Label Hypergraph Reasoner | label_hypergraph | 16 | **0.8684** | **0.8655** | 24 |

## 表 3. 输入实例数消融

该组实验固定最佳模块 `Label Hypergraph Reasoner`，比较每次检查采样图像数量对结果的影响。

| train max instances | Macro F1 | Micro F1 | Best epoch |
|---:|---:|---:|---:|
| 8 | 0.8582 | 0.8539 | 20 |
| 12 | 0.8600 | 0.8571 | 18 |
| 16 | **0.8684** | **0.8655** | 24 |
| 20 | 0.8572 | 0.8536 | 13 |
| 24 | 0.8506 | 0.8462 | 8 |

## 表 4. 最佳模型分标签结果

该表使用最终 `Label graph MIL` 的 `best_macro_f1` checkpoint。

| Label | Acc | Precision | Recall | Specificity | F1 | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 食管 SMT | 0.8258 | 0.7651 | 0.9203 | 0.7383 | 0.8355 | 0.8924 | 0.8557 |
| 食管黏膜/肿物病变 | 0.8345 | 0.8193 | 0.8767 | 0.7883 | 0.8470 | 0.9043 | 0.9257 |
| 胃炎 | 0.9303 | 0.9157 | 0.9300 | 0.9306 | 0.9228 | 0.9593 | 0.9484 |

## 可写入论文的实验结论

1. 在最终 11 个模型对比中，本文模型 `Label graph MIL` 在 macro F1、micro F1 和 kappa 上均取得最优结果，说明标签级证据聚合与标签关系建模有助于提升检查级多标签识别性能。
2. 在标签图推理模块消融中，移除标签图后 macro F1 从 `0.8598` 下降到 `0.8539`；进一步比较多种标签关系建模形式后，`Label Hypergraph Reasoner` 取得最高 macro F1 `0.8684`，提示高阶标签关系建模对该任务更有效。
3. 输入实例数消融显示，`train_max_instances=16` 时效果最佳；继续增加到 20 或 24 并未带来收益，可能是因为过多图像引入了更多与目标标签无关的背景帧。
4. 分标签结果显示，胃炎类别表现最好，F1 为 `0.9228`；两个食管相关标签的 F1 分别为 `0.8355` 和 `0.8470`，相对更具挑战性。

## 结果来源

| 内容 | 路径 |
|---|---|
| 自动探索汇总 | `/home/Lim/Project4/outputs/train_runs/task1/task1_ablation/exp_task1_auto_module_instance_search/module_instance_search_summary.md` |
| 最终模型对比 | `/home/Lim/Project4/outputs/train_runs/task1/task1_ablation/exp_task1_auto_module_instance_search/final_models_best_params/` |
| 最佳模型测试结果 | `/home/Lim/Project4/outputs/train_runs/task1/task1_ablation/exp_task1_auto_module_instance_search/final_models_best_params/train_001_final_label_graph_mil/test_result.csv` |
| TASK1 数据摘要 | `/home/Lim/Project4/datasets/task_data/task1/README.md` |
