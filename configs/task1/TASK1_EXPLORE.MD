# TASK1 自动探索输出说明

当前自动探索配置文件为：

- `configs/task1/auto_explore.yaml`

当 `configs/task1/train.yaml` 中的 `auto_explore: true` 时，执行 `python train.py` 会进入自动探索流程。

## 当前配置摘要

当前 `configs/task1/auto_explore.yaml` 的重点配置如下：

- `num_trials: 24`
- `trial_max_epochs: 30`
- `trial_patience: 10`
- `trial_run_test: false`
- `objective.name: stable_tail_val_loss`
- `stability_filter.max_final_gap: 0.05`
- `stability_filter.max_val_loss_rebound_ratio: 0.25`
- `optuna.pruner.n_warmup_steps: 5`

当前启用搜索的参数范围：

- `lr`：`1e-5 ~ 3e-4`
- `weight_decay`：`0.01 ~ 0.3`
- `warmup_ratio`：`0.05 ~ 0.25`
- `random_instance_dropout`：`0.05 ~ 0.3`

## 输出目录结构

所有训练输出统一放在：

- `train_runs/`

再按任务拆分为：

- `train_runs/task1/`
- `train_runs/task2/`

自动探索运行目录命名规则：

- TASK1：`task1_<运行次数>_para_auto`
- TASK2：`task2_<运行次数>_para_auto`

例如：

- `train_runs/task1/task1_1_para_auto/`

## 自动探索运行目录内容

每个自动探索运行目录下会包含：

- `notes.json`
- `remark.txt`
- `train_001/`
- `train_002/`
- `train_003/`

其中：

- `notes.json`：给机器读取的结构化摘要，包含每个训练目录的参数、验证选优结果、稳定性分析，以及在启用测试时的测试结果。
- `remark.txt`：给人看的简短总结，重点说明推荐训练目录、稳定候选和各训练目录概览。

## 单个训练目录内容

每个 `train_xxx/` 目录直接就是训练产物，不再额外套模型子目录，也不再在训练目录里写 `remark.txt`。

训练目录内主要包含：

- `config.yaml`
- `log.csv`
- `loss_curve.png`
- `last_confusion_matrix.png`
- `checkpoints/last.ckpt`
- `checkpoints/best_macro_f1.ckpt`
- `checkpoints/best_micro_f1.ckpt`
- `checkpoints/best_val_loss.ckpt`
- `best_macro_f1_val_confusion_matrix.png`
- `best_micro_f1_val_confusion_matrix.png`
- `best_val_loss_val_confusion_matrix.png`

其中三个根目录混淆矩阵分别对应三个“最佳验证 checkpoint”的验证集混淆矩阵：

- `best_macro_f1_val_confusion_matrix.png`
- `best_micro_f1_val_confusion_matrix.png`
- `best_val_loss_val_confusion_matrix.png`

这三个文件直接放在训练目录根部，不放在 `best_*` 子目录下。

若 `trial_run_test: true`，训练目录下还会额外出现：

- `test_macro_f1/`
- `test_micro_f1/`
- `test_val_loss/`
- `test_result.csv`
- `test_report.csv`

## 自动探索中的测试逻辑

当前自动探索默认跳过测试。

当前配置中：

- `trial_run_test: false`

因此每个 `train_xxx/` 训练结束后，默认只保留验证集选优结果，用于更快完成参数搜索，不会立刻生成测试目录与测试报表。

若后续改为 `trial_run_test: true`，则会针对以下三个最佳验证 checkpoint 分别做测试：

- `best_macro_f1`
- `best_micro_f1`
- `best_val_loss`

每次测试后的结果都会写入对应目录：

- `test_macro_f1/`
- `test_micro_f1/`
- `test_val_loss/`

目录内会保存各自的测试指标与测试图像结果，其中测试混淆矩阵文件名分别为：

- `best_macro_f1_test_confusion_matrix.png`
- `best_micro_f1_test_confusion_matrix.png`
- `best_val_loss_test_confusion_matrix.png`

## 当前用途

自动探索的目标仍然是先筛出：

- 收敛更稳定的参数组合；
- 验证集表现更可靠的训练目录；
- 后续值得继续正式训练或复现的候选结果。
- 在当前小样本配置下，更快淘汰明显过拟合或几乎不收敛的参数组合。
