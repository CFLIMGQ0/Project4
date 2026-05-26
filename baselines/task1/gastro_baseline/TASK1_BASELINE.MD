# TASK1 Baseline 说明

## 目录结构

- 胃镜 baseline 目录：`baselines/task1/gastro_baseline/`
- TASK2 当前三标签任务可复用本目录下按 `num_labels` 参数化的 baseline；`baselines/task2/` 仅作为兼容入口预留。
- 自动运行配置：`configs/task1/auto_baselines.yaml`
- 自动输出目录：`/home/Lim/Project4/outputs/train_runs/task1/baselines`

## 当前胃镜 baseline

1. `gastro_attention_mil_baseline`
   说明：标签级注意力聚合，作为最标准的多标签 MIL 对照。
2. `gastro_mean_pool_baseline`
   说明：实例特征均值池化，作为最简单的全局聚合基线。
3. `gastro_max_pool_baseline`
   说明：实例最大响应聚合，模拟“单关键帧主导”场景。
4. `gastro_transformer_mil_baseline`
   说明：先做实例间上下文建模，再做标签级聚合。
5. `gastro_topk_mil_baseline`
   说明：每个标签只聚合 top-k 关键实例，强调显著病灶片段。

## 自动运行规则

- 在 `configs/task1/train.yaml` 中将 `auto_baselines: true` 打开后，会读取 `configs/task1/auto_baselines.yaml`。
- 训练超参数默认继承 `configs/task1/train.yaml`。
- `auto_baselines.yaml` 中仅负责补充模型结构参数和单模型覆盖项。
- 每个模型训练完成后会立即执行测试。
- 运行目录固定为：
  - `train_001_gastro_attention_mil_baseline`
  - `train_002_gastro_mean_pool_baseline`
  - `train_003_gastro_max_pool_baseline`
  - `train_004_gastro_transformer_mil_baseline`
  - `train_005_gastro_topk_mil_baseline`

## 断点续跑规则

- 如果某个模型目录已经存在完整测试结果，则自动跳过，不会重复训练。
- 如果目录中存在 `checkpoints/last.ckpt` 但测试未完成，则自动从该断点继续训练。
- 如果目录存在残留文件但没有 `last.ckpt`，则认为上次训练未形成可续跑断点，会清理该目录并重新训练。

## 输出文件

- `remark.txt`：整体对比摘要
- `notes.json`：结构化汇总，便于程序读取
- `log.csv`：每个 epoch 的训练/验证日志
- `test_result.csv`：当前模型各最佳 checkpoint 的测试结果
- `checkpoints/last.ckpt`：断点续训入口

## 常用命令

只运行 baseline 的推荐方式：

```bash
python train.py
```

前提：

- `configs/task1/train.yaml` 中设置 `auto_baselines: true`
- `configs/task1/train.yaml` 中设置 `auto_sotas: false`
