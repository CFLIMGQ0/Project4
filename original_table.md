# exp3–exp8 原始实验结果与存储盘点

本文档归档当前论文未直接采用的 exp3–exp8 实验结果，并统计相关训练产物的磁盘占用，供后续决定是否删除模型权重或整个实验目录。统计时间为2026年8月1日，所有文件均未删除。

## 统计口径

- 结果统一读取各训练目录 `test_result.csv` 中的 `best_macro_f1` 行，即使用验证集 Macro F1 选出的 checkpoint 在测试集上的结果。
- `exp3` 是旧版8标签任务；`exp4–exp8` 是当前3标签任务，两类结果不可直接横向比较。
- “权重”包括 checkpoint 目录及 `.ckpt`、`.pt`、`.pth`、`.bin`、`.safetensors` 文件。
- 磁盘占用按文件系统实际分配块统计，因此比 `du -h` 的整数显示更精确。
- `exp_mm_ablation_hypergraph` 是 exp8 的17组结构化字段消融，虽然物理目录名不含 `exp_8`，仍作为 exp8 关联实验纳入。

## 存储占用汇总

| 实验 | 训练目录数 | 完成/总数 | 文件数 | 总占用 | 权重占用 | 非权重占用 | `best_macro_f1`权重 | 其余权重（候选清理） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp3 | 11 | 10/11 | 283 | 12.15 GiB | 12.11 GiB | 0.040 GiB | 10个 / 3.03 GiB | 30个 / 9.08 GiB |
| exp4 | 7 | 7/7 | 198 | 9.69 GiB | 9.68 GiB | 0.013 GiB | 7个 / 2.42 GiB | 21个 / 7.26 GiB |
| exp5 | 1 | 1/1 | 30 | 1.52 GiB | 1.52 GiB | 0.002 GiB | 1个 / 0.38 GiB | 3个 / 1.14 GiB |
| exp6 | 12 | 7/12 | 203 | 10.69 GiB | 10.68 GiB | 0.011 GiB | 7个 / 2.67 GiB | 21个 / 8.01 GiB |
| exp7 | 7 | 7/7 | 198 | 10.66 GiB | 10.65 GiB | 0.013 GiB | 7个 / 2.66 GiB | 21个 / 7.98 GiB |
| exp8主实验 | 6 | 6/6 | 190 | 9.92 GiB | 9.91 GiB | 0.010 GiB | 6个 / 2.48 GiB | 18个 / 7.43 GiB |
| exp8结构化字段消融 | 17 | 17/17 | 545 | 27.31 GiB | 27.28 GiB | 0.031 GiB | 17个 / 6.82 GiB | 51个 / 20.46 GiB |
| **合计** | **61** | **55/61** | **1647** | **81.95 GiB** | **81.83 GiB** | **0.119 GiB** | **55个 / 20.46 GiB** | **165个 / 61.38 GiB** |

每个已完成训练通常保存4份同等大小的权重：`best_macro_f1.ckpt`、`best_micro_f1.ckpt`、`best_val_loss.ckpt` 和 `last.ckpt`。55个已完成训练共保存220份权重。

### 可选清理口径

| 方案 | 预计释放空间 | 预计保留空间 | 影响 |
|---|---:|---:|---|
| 仅删除其余3类权重，保留 `best_macro_f1.ckpt` | 61.38 GiB | 约20.58 GiB | 保留论文统一口径的最佳权重，可继续推理；不能从 `last.ckpt` 续训，也不能复测另外两种 checkpoint |
| 删除全部权重，仅保留配置、日志、结果CSV/JSON和图表 | 81.83 GiB | 约0.119 GiB | 可保留数值审计记录，但无法直接推理、复测或续训 |
| 删除全部 exp3–exp8 相关输出 | 81.95 GiB | 0 GiB | 仅保留本文档中的汇总数值；详细日志、逐标签结果和训练配置也会丢失 |

上述空间不包含 `src/exp_1`、`src/exp_2`、`src/exp_4`、`src/exp_5`、`src/exp_6`、`src/exp_8` 等源代码。即使清理输出，也建议保留源代码，尤其是 exp9 和现有 checkpoint 加载仍会依赖的 exp8 模型实现。

## exp3：旧版8标签实验

输出目录：`/home/Lim/Project4/outputs/train_runs/task2/exp_3`

| 训练目录 | 模型名 | 状态 | Best epoch | Macro F1 | Micro F1 | Macro ROC-AUC | Macro PR-AUC | Subset Accuracy | Hamming Loss | Kappa |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `auto_exp_3/train_001_ot_mil` | `ot_mil` | 完成 | 25 | 0.4598 | 0.6489 | 0.8023 | 0.4426 | 0.2850 | 0.2053 | 0.5126 |
| `auto_exp_3/train_002_ib_mil` | `ib_mil` | 完成 | 16 | 0.4541 | 0.5831 | 0.8095 | 0.4677 | 0.1762 | 0.2742 | 0.4044 |
| `auto_exp_3/train_003_hyp_mil` | `hyp_mil` | 完成 | 29 | 0.4588 | 0.6710 | 0.8004 | 0.4561 | 0.3506 | 0.1839 | 0.5486 |
| `auto_exp_3/train_004_sd_mil` | `sd_mil` | 完成 | 15 | 0.4586 | 0.5909 | 0.8221 | 0.4609 | 0.2211 | 0.2712 | 0.4145 |
| `auto_exp_3/train_005_edl_mil` | `edl_mil` | 完成 | 16 | 0.4275 | 0.7357 | 0.8340 | 0.4621 | 0.4680 | 0.1250 | 0.6539 |
| `auto_exp_3/train_006_db_mil` | `db_mil` | 完成 | 21 | 0.3946 | 0.6734 | 0.8070 | 0.4434 | 0.4611 | 0.1388 | 0.5862 |
| `auto_exp_3/train_007_aslt_mil` | `aslt_mil` | **未完成** | — | — | — | — | — | — | — | — |
| `auto_exp_3/train_008_laca_mil` | `laca_mil` | 完成 | 21 | 0.4726 | 0.7131 | 0.8130 | 0.4607 | 0.4421 | 0.1505 | 0.6130 |
| `auto_exp_3/train_009_cl_mil` | `cl_mil` | 完成 | 10 | 0.3772 | 0.7256 | 0.8267 | 0.4405 | 0.4888 | 0.1287 | 0.6416 |
| `auto_exp_3/train_010_csml_mil` | `csml_mil` | 完成 | 14 | 0.4077 | 0.7032 | 0.8257 | 0.4466 | 0.4594 | 0.1416 | 0.6103 |
| `task2_1` | `gastro_label_graph_mil` | 完成 | 14 | 0.4716 | 0.6877 | 0.8286 | 0.4735 | 0.4007 | 0.1710 | 0.5738 |

## exp4：三标签图像MIL结构探索

输出目录：`/home/Lim/Project4/outputs/train_runs/task2/exp_4`

| 训练目录 | 模型名 | 状态 | Best epoch | Macro F1 | Micro F1 | Macro ROC-AUC | Macro PR-AUC | Subset Accuracy | Hamming Loss | Kappa |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `auto_exp_4/train_001_gastro_label_graph_mil` | `gastro_label_graph_mil` | 完成 | 16 | 0.8547 | 0.8508 | 0.9190 | 0.9082 | 0.6906 | 0.1493 | 0.7021 |
| `auto_exp_4/train_002_multi_sample_lg_mil` | `multi_sample_lg_mil` | 完成 | 12 | 0.8590 | 0.8546 | 0.9236 | 0.9148 | 0.6581 | 0.1493 | 0.7028 |
| `auto_exp_4/train_003_full_feature_mil` | `full_feature_mil` | 完成 | 15 | 0.8567 | 0.8534 | 0.9277 | 0.9169 | 0.7026 | 0.1453 | 0.7098 |
| `auto_exp_4/train_004_hier_full_mil` | `hier_full_mil` | 完成 | 18 | 0.8664 | 0.8626 | 0.9268 | 0.9147 | 0.6923 | 0.1385 | 0.7239 |
| `auto_exp_4/train_005_hier_full_lg_mil` | `hier_full_lg_mil` | 完成 | 20 | 0.8629 | 0.8599 | 0.9242 | 0.9116 | 0.6991 | 0.1402 | 0.7203 |
| `auto_exp_4/train_006_long_mil` | `long_mil` | 完成 | 22 | 0.8759 | 0.8723 | 0.9289 | 0.9139 | 0.7436 | 0.1265 | 0.7473 |
| `auto_exp_4/train_007_mamba_mil` | `mamba_mil` | 完成 | 21 | 0.8628 | 0.8605 | 0.9198 | 0.9037 | 0.7350 | 0.1362 | 0.7276 |

## exp5：ROI Long-MIL实验

输出目录：`/home/Lim/Project4/outputs/train_runs/task2/exp_5`

| 训练目录 | 模型名 | 状态 | Best epoch | Macro F1 | Micro F1 | Macro ROC-AUC | Macro PR-AUC | Subset Accuracy | Hamming Loss | Kappa |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `auto_exp_5/train_001_roi_long_mil` | `long_mil` | 完成 | 23 | 0.8741 | 0.8712 | 0.9287 | 0.9112 | 0.7162 | 0.1288 | 0.7430 |

## exp6：ROI输入组织与双路模型探索

输出目录：`/home/Lim/Project4/outputs/train_runs/task2/exp_6`

| 训练目录 | 模型名 | 状态 | Best epoch | Macro F1 | Micro F1 | Macro ROC-AUC | Macro PR-AUC | Subset Accuracy | Hamming Loss | Kappa |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `auto_exp_6/train_001_exp6_long_mil_64_no_roi` | `long_mil` | 完成 | 10 | 0.8705 | 0.8671 | 0.9269 | 0.9151 | 0.7299 | 0.1328 | 0.7350 |
| `auto_exp_6/train_001_exp6_roi_context_128_16` | `long_mil` | 完成 | 18 | 0.8806 | 0.8776 | 0.9311 | 0.9137 | 0.7487 | 0.1208 | 0.7586 |
| `auto_exp_6/train_002_exp6_roi_dual_128_16` | `exp6_dual_stream_long_mil` | 完成 | 22 | 0.8595 | 0.8557 | 0.9240 | 0.9149 | 0.7026 | 0.1447 | 0.7112 |
| `auto_exp_6/train_002_exp6_roi_mix_64_32` | `long_mil` | 完成 | 28 | 0.8719 | 0.8685 | 0.9309 | 0.9129 | 0.7282 | 0.1299 | 0.7404 |
| `auto_exp_6/train_003_exp6_roi_mix_64_64` | `long_mil` | 完成 | 17 | 0.8764 | 0.8731 | 0.9292 | 0.9164 | 0.7436 | 0.1259 | 0.7485 |
| `auto_exp_6/train_005_exp6_roi_context_128_32` | `long_mil` | **未完成** | — | — | — | — | — | — | — | — |
| `auto_exp_6/train_006_exp6_roi_context_128_64` | `long_mil` | **未完成** | — | — | — | — | — | — | — | — |
| `auto_exp_6/train_008_exp6_roi_dual_128_32` | `exp6_dual_stream_long_mil` | **未完成** | — | — | — | — | — | — | — | — |
| `auto_exp_6/train_009_exp6_roi_dual_128_64` | `exp6_dual_stream_long_mil` | **未完成** | — | — | — | — | — | — | — | — |
| `auto_exp_6/train_010_exp6_roi_filter_96_32` | `exp6_dual_stream_long_mil` | 完成 | 18 | 0.8695 | 0.8664 | 0.9297 | 0.9196 | 0.7470 | 0.1316 | 0.7370 |
| `auto_exp_6/train_011_exp6_roi_filter_128_32` | `exp6_dual_stream_long_mil` | 完成 | 24 | 0.8780 | 0.8756 | 0.9311 | 0.9203 | 0.7333 | 0.1242 | 0.7521 |
| `auto_exp_6/train_012_exp6_roi_cons_128_32` | `exp6_dual_stream_long_mil` | **未完成** | — | — | — | — | — | — | — | — |

## exp7：ROI mix与ROI context对照

输出目录：`/home/Lim/Project4/outputs/train_runs/task2/exp_7`

| 训练目录 | 模型名 | 状态 | Best epoch | Macro F1 | Micro F1 | Macro ROC-AUC | Macro PR-AUC | Subset Accuracy | Hamming Loss | Kappa |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `auto_exp_7/train_001_exp6_long_mil_64_no_roi` | `long_mil` | 完成 | 27 | 0.8838 | 0.8807 | 0.9308 | 0.9194 | 0.7641 | 0.1168 | 0.7664 |
| `auto_exp_7/train_002_exp6_roi_mix_64_16` | `long_mil` | 完成 | 20 | 0.8669 | 0.8633 | 0.9285 | 0.9164 | 0.7350 | 0.1345 | 0.7312 |
| `auto_exp_7/train_003_exp6_roi_mix_128_16` | `long_mil` | 完成 | 11 | 0.8695 | 0.8665 | 0.9249 | 0.9115 | 0.7145 | 0.1345 | 0.7318 |
| `auto_exp_7/train_004_exp6_roi_context_64_16` | `long_mil` | 完成 | 23 | 0.8665 | 0.8629 | 0.9267 | 0.9135 | 0.7231 | 0.1356 | 0.7291 |
| `auto_exp_7/train_005_exp6_roi_context_64_32` | `long_mil` | 完成 | 27 | 0.8698 | 0.8666 | 0.9283 | 0.9113 | 0.7368 | 0.1305 | 0.7391 |
| `auto_exp_7/train_006_exp6_roi_context_64_64` | `long_mil` | 完成 | 20 | 0.8617 | 0.8578 | 0.9257 | 0.9137 | 0.7094 | 0.1413 | 0.7178 |
| `auto_exp_7/train_007_exp6_long_mil_128_no_roi` | `long_mil` | 完成 | 19 | 0.8693 | 0.8658 | 0.9293 | 0.9157 | 0.7231 | 0.1333 | 0.7337 |

## exp8：多模态融合主实验

输出目录：`/home/Lim/Project4/outputs/train_runs/task2/exp_8`

| 训练目录 | 模型名 | 状态 | Best epoch | Macro F1 | Micro F1 | Macro ROC-AUC | Macro PR-AUC | Subset Accuracy | Hamming Loss | Kappa |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `train_001_exp8_mm_struct_late_gate` | `exp8_mm_struct_late_gate` | 完成 | 29 | 0.8719 | 0.8698 | 0.9356 | 0.9247 | 0.7496 | 0.1258 | 0.7483 |
| `train_002_exp8_mm_label_proto_graph` | `exp8_mm_label_proto_graph` | 完成 | 19 | 0.8787 | 0.8773 | 0.9366 | 0.9226 | 0.7513 | 0.1205 | 0.7594 |
| `train_003_exp8_mm_text_contrast_distill` | `exp8_mm_text_contrast_distill` | 完成 | 29 | 0.8777 | 0.8757 | 0.9389 | 0.9295 | 0.7478 | 0.1217 | 0.7569 |
| `train_004_exp8_mm_watch_cross_attn` | `exp8_mm_watch_cross_attn` | 完成 | 16 | 0.8871 | 0.8856 | 0.9387 | 0.9154 | 0.7672 | 0.1123 | 0.7758 |
| `train_005_exp8_mm_text_guided_top64_align` | `exp8_mm_text_guided_top64_align` | 完成 | 19 | 0.8807 | 0.8794 | 0.9365 | 0.9160 | 0.7584 | 0.1176 | 0.7650 |
| `train_007_exp8_mm_ablation_title_operation` | `exp8_structured_late_gate_mil` | 完成 | 29 | 0.8801 | 0.8784 | 0.9376 | 0.9243 | 0.7743 | 0.1170 | 0.7658 |

## exp8关联：结构化字段消融

输出目录：`/home/Lim/Project4/outputs/train_runs/task2/exp_mm_ablation_hypergraph`

| 训练目录 | 模型名 | 状态 | Best epoch | Macro F1 | Micro F1 | Macro ROC-AUC | Macro PR-AUC | Subset Accuracy | Hamming Loss | Kappa |
|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `train_001_exp8_mm_ablation_image_baseline` | `exp8_structured_late_gate_mil` | 完成 | 26 | 0.8719 | 0.8701 | 0.9356 | 0.9218 | 0.7478 | 0.1258 | 0.7484 |
| `train_002_exp8_mm_ablation_age` | `exp8_structured_late_gate_mil` | 完成 | 20 | 0.8720 | 0.8698 | 0.9367 | 0.9232 | 0.7513 | 0.1258 | 0.7483 |
| `train_003_exp8_mm_ablation_age_sex` | `exp8_structured_late_gate_mil` | 完成 | 10 | 0.8738 | 0.8715 | 0.9378 | 0.9244 | 0.7654 | 0.1235 | 0.7528 |
| `train_004_exp8_mm_ablation_age_sex_hp` | `exp8_structured_late_gate_mil` | 完成 | 20 | 0.8713 | 0.8689 | 0.9353 | 0.9234 | 0.7443 | 0.1276 | 0.7450 |
| `train_005_exp8_mm_ablation_reportTitle` | `exp8_structured_late_gate_mil` | 完成 | 28 | 0.8773 | 0.8758 | 0.9359 | 0.9223 | 0.7549 | 0.1205 | 0.7590 |
| `train_006_exp8_mm_ablation_operationValue` | `exp8_structured_late_gate_mil` | 完成 | 16 | 0.8757 | 0.8737 | 0.9349 | 0.9223 | 0.7496 | 0.1229 | 0.7544 |
| `train_007_exp8_mm_ablation_title_operation` | `exp8_structured_late_gate_mil` | 完成 | 17 | 0.8761 | 0.8745 | 0.9388 | 0.9233 | 0.7619 | 0.1217 | 0.7566 |
| `train_008_exp8_mm_ablation_all_structured` | `exp8_structured_late_gate_mil` | 完成 | 17 | 0.8766 | 0.8746 | 0.9381 | 0.9245 | 0.7601 | 0.1217 | 0.7567 |
| `train_009_exp8_mm_ablation_all_without_title` | `exp8_structured_late_gate_mil` | 完成 | 10 | 0.8788 | 0.8765 | 0.9406 | 0.9279 | 0.7619 | 0.1199 | 0.7602 |
| `train_010_exp8_mm_ablation_all_without_operation` | `exp8_structured_late_gate_mil` | 完成 | 28 | 0.8761 | 0.8745 | 0.9362 | 0.9232 | 0.7531 | 0.1217 | 0.7566 |
| `train_011_exp8_mm_ablation_all_without_hp` | `exp8_structured_late_gate_mil` | 完成 | 10 | 0.8823 | 0.8800 | 0.9391 | 0.9262 | 0.7690 | 0.1164 | 0.7672 |
| `train_012_exp8_mm_ablation_all_without_age` | `exp8_structured_late_gate_mil` | 完成 | 28 | 0.8807 | 0.8789 | 0.9360 | 0.9215 | 0.7725 | 0.1170 | 0.7659 |
| `train_013_exp8_mm_ablation_all_shuffle_title_test` | `exp8_structured_late_gate_mil` | 完成 | 17 | 0.8763 | 0.8745 | 0.9381 | 0.9242 | 0.7619 | 0.1217 | 0.7566 |
| `train_014_exp8_mm_ablation_all_shuffle_operation_test` | `exp8_structured_late_gate_mil` | 完成 | 17 | 0.8747 | 0.8729 | 0.9378 | 0.9238 | 0.7566 | 0.1235 | 0.7531 |
| `train_015_exp8_mm_ablation_all_shuffle_title_operation_test` | `exp8_structured_late_gate_mil` | 完成 | 17 | 0.8758 | 0.8739 | 0.9374 | 0.9225 | 0.7601 | 0.1223 | 0.7555 |
| `train_016_exp8_mm_ablation_shuffle_title_train` | `exp8_structured_late_gate_mil` | 完成 | 17 | 0.8781 | 0.8761 | 0.9376 | 0.9253 | 0.7601 | 0.1205 | 0.7591 |
| `train_017_exp8_mm_ablation_shuffle_operation_train` | `exp8_structured_late_gate_mil` | 完成 | 10 | 0.8829 | 0.8810 | 0.9374 | 0.9245 | 0.7637 | 0.1158 | 0.7685 |

## 未完成实验目录

以下6个目录只有配置文件，没有测试结果，也没有有效 checkpoint，磁盘占用可以忽略：

- `exp_3/auto_exp_3/train_007_aslt_mil`
- `exp_6/auto_exp_6/train_005_exp6_roi_context_128_32`
- `exp_6/auto_exp_6/train_006_exp6_roi_context_128_64`
- `exp_6/auto_exp_6/train_008_exp6_roi_dual_128_32`
- `exp_6/auto_exp_6/train_009_exp6_roi_dual_128_64`
- `exp_6/auto_exp_6/train_012_exp6_roi_cons_128_32`

## 删除决策提示

- 当前论文 `main.tex` 未直接使用上述实验数值；论文主结果使用 exp9、exp10、exp11，以及独立的图像 baseline/SOTA 目录。
- 如果只希望快速给 TASK3 腾空间，同时保留所有最佳模型，删除165个非 `best_macro_f1` 权重即可释放约61.38 GiB。
- 如果确认以后不再对 exp3–exp8 做推理或复测，可以删除全部220个权重，释放约81.83 GiB，同时保留体积很小的配置、日志和结果文件。
- 不建议因为清理输出而删除 `src/exp_8` 等源代码；现有后续模型仍可能引用这些类定义。
