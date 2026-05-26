# TASK2 训练结果汇总

- 整理时间：2026-05-25 15:20:06 CST
- 输出目录：`/home/Lim/Project4/outputs/train_runs/task2/exp_mm_ablation_hypergraph`
- 当前配置：`configs/task2/train.yaml` 中 `auto_exp_8_mm_ablation: true`，本轮是 TASK2 的 exp_8 多模态短字段消融。
- 完成情况：已配置 17 个实验；已产出测试结果 15 个；未产出测试结果 2 个。
- 运行状态：检测到 `python train.py` 仍在运行，最长已运行约 4.32 天，估计启动时间 2026-05-21 07:33:56 CST。
- 当前测试集最高 `macro_f1`：`图像 + reportTitle + operationValue`（`train_007_exp8_mm_ablation_title_operation`），`macro_f1=0.8841`，`micro_f1=0.8824`，`best_epoch=15`。
- 说明：下表均取每个实验 `test_result.csv` 中 `checkpoint_alias=best_macro_f1` 的测试结果。

## 主指标表

| 序号 | 实验目录 | 实验设置 | best_epoch | test_loss | macro_f1 | 相对图像基线 | micro_f1 | macro_auc | macro_ap | subset_acc | hamming_loss | kappa |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `train_001_exp8_mm_ablation_image_baseline` | 图像 Long-MIL 64 原图基线 | 26 | 0.1521 | 0.8709 | +0.0000 | 0.8691 | 0.9355 | 0.9210 | 0.7443 | 0.1270 | 0.7461 |
| 2 | `train_002_exp8_mm_ablation_age` | 图像 + age | 13 | 0.1366 | 0.8729 | +0.0020 | 0.8701 | 0.9386 | 0.9266 | 0.7496 | 0.1258 | 0.7484 |
| 3 | `train_003_exp8_mm_ablation_age_sex` | 图像 + age + sex | 10 | 0.1180 | 0.8733 | +0.0024 | 0.8709 | 0.9378 | 0.9247 | 0.7654 | 0.1240 | 0.7517 |
| 4 | `train_004_exp8_mm_ablation_age_sex_hp` | 图像 + age + sex + hp | 16 | 0.1416 | 0.8716 | +0.0006 | 0.8691 | 0.9347 | 0.9223 | 0.7337 | 0.1282 | 0.7440 |
| 5 | `train_005_exp8_mm_ablation_reportTitle` | 图像 + reportTitle | 21 | 0.1523 | 0.8751 | +0.0041 | 0.8736 | 0.9356 | 0.9218 | 0.7513 | 0.1229 | 0.7543 |
| 6 | `train_006_exp8_mm_ablation_operationValue` | 图像 + operationValue | 29 | 0.1584 | 0.8800 | +0.0091 | 0.8789 | 0.9350 | 0.9222 | 0.7672 | 0.1170 | 0.7659 |
| 7 | `train_007_exp8_mm_ablation_title_operation` | 图像 + reportTitle + operationValue | 15 | 0.1393 | 0.8841 | +0.0131 | 0.8824 | 0.9402 | 0.9274 | 0.7848 | 0.1135 | 0.7730 |
| 8 | `train_008_exp8_mm_ablation_all_structured` | 图像 + 全部结构化字段 | 13 | 0.1352 | 0.8710 | +0.0001 | 0.8684 | 0.9371 | 0.9257 | 0.7425 | 0.1282 | 0.7438 |
| 9 | `train_009_exp8_mm_ablation_all_without_title` | 全字段去掉 reportTitle | 15 | 0.1382 | 0.8836 | +0.0127 | 0.8818 | 0.9409 | 0.9296 | 0.7901 | 0.1135 | 0.7728 |
| 10 | `train_010_exp8_mm_ablation_all_without_operation` | 全字段去掉 operationValue | 26 | 0.1536 | 0.8697 | -0.0013 | 0.8676 | 0.9358 | 0.9242 | 0.7460 | 0.1282 | 0.7436 |
| 11 | `train_011_exp8_mm_ablation_all_without_hp` | 全字段去掉 hp | 10 | 0.1085 | 0.8771 | +0.0062 | 0.8749 | 0.9389 | 0.9254 | 0.7566 | 0.1217 | 0.7567 |
| 12 | `train_012_exp8_mm_ablation_all_without_age` | 全字段去掉 age | 13 | 0.1282 | 0.8766 | +0.0057 | 0.8743 | 0.9366 | 0.9229 | 0.7460 | 0.1235 | 0.7535 |
| 13 | `train_013_exp8_mm_ablation_all_shuffle_title_test` | 全字段训练，测试集置乱 reportTitle | 13 | 0.1350 | 0.8703 | -0.0007 | 0.8677 | 0.9370 | 0.9246 | 0.7425 | 0.1287 | 0.7426 |
| 14 | `train_014_exp8_mm_ablation_all_shuffle_operation_test` | 全字段训练，测试集置乱 operationValue | 13 | 0.1347 | 0.8703 | -0.0006 | 0.8677 | 0.9366 | 0.9247 | 0.7425 | 0.1287 | 0.7426 |
| 15 | `train_015_exp8_mm_ablation_all_shuffle_title_operation_test` | 全字段训练，测试集同时置乱 reportTitle 与 operationValue | 13 | 0.1348 | 0.8719 | +0.0009 | 0.8689 | 0.9365 | 0.9232 | 0.7443 | 0.1276 | 0.7450 |

## 按标签 F1 表

| 序号 | 实验设置 | 食管 SMT F1 | 食管黏膜/肿瘤 F1 | 胃炎 F1 | macro_recall | macro_precision | macro_specificity |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 图像 Long-MIL 64 原图基线 | 0.8421 | 0.8658 | 0.9049 | 0.9031 | 0.8416 | 0.8471 |
| 2 | 图像 + age | 0.8372 | 0.8654 | 0.9161 | 0.9030 | 0.8453 | 0.8486 |
| 3 | 图像 + age + sex | 0.8427 | 0.8650 | 0.9122 | 0.8975 | 0.8514 | 0.8587 |
| 4 | 图像 + age + sex + hp | 0.8396 | 0.8659 | 0.9091 | 0.9091 | 0.8370 | 0.8338 |
| 5 | 图像 + reportTitle | 0.8473 | 0.8730 | 0.9049 | 0.9087 | 0.8443 | 0.8494 |
| 6 | 图像 + operationValue | 0.8592 | 0.8766 | 0.9042 | 0.9103 | 0.8533 | 0.8627 |
| 7 | 图像 + reportTitle + operationValue | 0.8534 | 0.8786 | 0.9202 | 0.9124 | 0.8584 | 0.8653 |
| 8 | 图像 + 全部结构化字段 | 0.8345 | 0.8668 | 0.9118 | 0.9053 | 0.8403 | 0.8423 |
| 9 | 全字段去掉 reportTitle | 0.8493 | 0.8791 | 0.9224 | 0.9087 | 0.8616 | 0.8713 |
| 10 | 全字段去掉 operationValue | 0.8375 | 0.8662 | 0.9052 | 0.8982 | 0.8432 | 0.8474 |
| 11 | 全字段去掉 hp | 0.8389 | 0.8774 | 0.9149 | 0.9120 | 0.8460 | 0.8502 |
| 12 | 全字段去掉 age | 0.8400 | 0.8696 | 0.9202 | 0.9190 | 0.8382 | 0.8374 |
| 13 | 全字段训练，测试集置乱 reportTitle | 0.8345 | 0.8668 | 0.9095 | 0.9038 | 0.8401 | 0.8423 |
| 14 | 全字段训练，测试集置乱 operationValue | 0.8351 | 0.8663 | 0.9095 | 0.9040 | 0.8403 | 0.8425 |
| 15 | 全字段训练，测试集同时置乱 reportTitle 与 operationValue | 0.8330 | 0.8668 | 0.9158 | 0.9053 | 0.8419 | 0.8431 |

## 未完成实验状态

| 序号 | 实验目录 | 实验设置 | 状态 | 最新验证 epoch | 最新 val_loss | 最新 val macro_f1 | 最佳验证 macro_f1(epoch) |
|---:|---|---|---|---:|---:|---:|---:|
| 16 | `train_016_exp8_mm_ablation_shuffle_title_train` | 全字段训练/验证/测试均置乱 reportTitle | 运行中/待测试 | 29 | 0.1476 | 0.8813 | 0.8851(17) |
| 17 | `-` | 全字段训练/验证/测试均置乱 operationValue | 未开始 | - | - | - | - |

## 结果观察

- 当前最优测试结果来自 `train_007_exp8_mm_ablation_title_operation`：只加入 `reportTitle + operationValue` 的组合，`macro_f1=0.8841`，比图像基线高 `+0.0131`。
- 单字段对比里，`operationValue` 的提升更明显：`图像 + operationValue` 的 `macro_f1=0.8800`，高于 `图像 + reportTitle` 的 `0.8751`。
- `全字段去掉 reportTitle` 仍达到 `macro_f1=0.8836`，但 `全字段去掉 operationValue` 降到 `0.8697`，说明当前实验里 `operationValue` 对测试指标更关键。
- 测试集置乱 `reportTitle`、`operationValue` 或二者同时置乱后，`macro_f1` 均约为 `0.8703-0.8719`，接近图像基线，低于未置乱的标题+操作字段组合。
- `notes.json/remark.txt` 对已完成的 15 个实验均标记为 `stable_convergence=false`，原因是 `train/val loss` 差距偏大且验证损失后期反弹；因此建议优先使用 best checkpoint 的测试结果，不建议用最后 epoch 作为结论。

## 进程快照

| PID | 已运行秒数 | CPU% | MEM% | 状态 | 命令 |
|---:|---:|---:|---:|---|---|
| 3018051 | 475 | 93.3 | 5.4 | `Rl+` | `/home/Lim/anaconda3/envs/myenv/bin/python train.py` |
| 3018052 | 475 | 93.1 | 5.3 | `Rl+` | `/home/Lim/anaconda3/envs/myenv/bin/python train.py` |
| 3186390 | 373570 | 0.1 | 0.0 | `Sl+` | `python train.py` |
| 3186391 | 373570 | 81.6 | 6.5 | `Sl+` | `/home/Lim/anaconda3/envs/myenv/bin/python train.py` |
