# 自动探索输出说明

当前自动探索配置文件为：

- `configs/auto_explore.yaml`

当 `configs/train.yaml` 中的 `auto_explore: true` 时，执行 `python train.py` 会进入自动探索流程。

## 输出目录结构

所有训练输出统一放在：

- `train_runs/`

再按任务拆分为：

- `train_runs/gastro_multilabel_task/`
- `train_runs/colonoscopy_binary_task/`

自动探索运行目录命名规则：

- 胃镜：`gastro_<运行次数>_para_auto`
- 肠镜：`colonoscopy_<运行次数>_para_auto`

例如：

- `train_runs/gastro_multilabel_task/gastro_1_para_auto/`

## 自动探索运行目录内容

每个自动探索运行目录下会包含：

- `notes.json`
- `remark.txt`
- `train_001/`
- `train_002/`
- `train_003/`

其中：

- `notes.json`：给机器读取的结构化摘要，包含每个训练目录的参数、验证选优结果、测试结果和稳定性分析。
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
- `test_macro_f1/`
- `test_micro_f1/`
- `test_val_loss/`
- `best_macro_f1_val_confusion_matrix.png`
- `best_micro_f1_val_confusion_matrix.png`
- `best_val_loss_val_confusion_matrix.png`
- `test_result.csv`
- `test_report.csv`

其中三个根目录混淆矩阵分别对应三个“最佳验证 checkpoint”的验证集混淆矩阵：

- `best_macro_f1_val_confusion_matrix.png`
- `best_micro_f1_val_confusion_matrix.png`
- `best_val_loss_val_confusion_matrix.png`

这三个文件直接放在训练目录根部，不放在 `best_*` 子目录下。

## 自动探索中的测试逻辑

自动探索不再跳过测试。

每个 `train_xxx/` 训练结束后，会立刻针对以下三个最佳验证 checkpoint 分别做测试：

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
