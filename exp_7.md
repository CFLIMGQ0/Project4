# exp_7 实验计划

`exp_7` 面向 TASK2 胃镜检查级三标签多标签分类，目标是在 `exp_6` 结果基础上，进一步判断加入 ROI 弱监督后，`ROI mix` 与 `ROI context` 两类 Long-MIL 输入组织方式哪一种更稳。

## 运行方式

当前代码已接入 `auto_exp_7`，默认配置位于 `configs/task2/train.yaml`。在项目 `src` 目录下直接运行：

```bash
python train.py
```

输出目录为：

```text
outputs/train_runs/task2/exp_7/
└── auto_exp_7/
    ├── train_001_exp6_long_mil_64_no_roi/
    ├── train_002_exp6_roi_mix_64_16/
    ├── train_003_exp6_roi_mix_128_16/
    ├── train_004_exp6_roi_context_64_16/
    ├── train_005_exp6_roi_context_64_32/
    └── train_006_exp6_roi_context_64_64/
```

## 实验列表

用户计划里写的是“总共五个实验”，但具体条目包含 `1` 个 no-ROI 对照和 `5` 个 ROI 变体，因此当前按列出的全部 `6` 个训练目录执行。

| 序号 | 实验名 | 基础模型 | 输入组织 | 目的 |
|---:|---|---|---|---|
| 1 | `exp6_long_mil_64_no_roi` | `long_mil` | `64` 原图，无 ROI | no-ROI 公平对照 |
| 2 | `exp6_roi_mix_64_16` | `long_mil` | `64` 原图 + `16` ROI | 小 ROI 数量的 mix 变体 |
| 3 | `exp6_roi_mix_128_16` | `long_mil` | `128` 原图 + `16` ROI | 更多原图上下文的 mix 变体 |
| 4 | `exp6_roi_context_64_16` | `long_mil` | `64` 原图 + `16` ROI | context 分组的小 ROI 数量变体 |
| 5 | `exp6_roi_context_64_32` | `long_mil` | `64` 原图 + `32` ROI | context 分组的中等 ROI 数量变体 |
| 6 | `exp6_roi_context_64_64` | `long_mil` | `64` 原图 + `64` ROI | context 分组的大 ROI 数量变体 |

## 评价方式

- 每个训练目录会保存 `config.yaml`、`log.csv`、`test_result.csv`、`test_report.csv`、`test_micro_f1/`、`test_macro_f1/`、`test_val_loss/` 等产物。
- `auto_exp_7` 汇总时默认使用 `best_macro_f1` checkpoint 的测试集 `macro_f1` 作为主排序指标。
- 每次训练结束后，`auto_exp_7/notes.json` 和 `auto_exp_7/remark.txt` 会记录最佳模型、所有模型指标和训练稳定性评价。

## 实现说明

当前 `ROI mix` 与 `ROI context` 都复用 `long_mil`，差异通过 `run_overrides` 控制：

- `train_max_instances = 原图数量 + ROI 数量`
- `roi_max_crops_per_bag = ROI 数量`
- 数据加载时会先保留最多 `train_max_instances - roi_max_crops_per_bag` 张原图，再追加 ROI crop。

也就是说，本轮实验先把不同原图数量和 ROI 数量组合跑齐，用统一测试指标决定下一步是否需要引入更明确的结构差异。
