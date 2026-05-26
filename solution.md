# TASK2 三标签训练优化与类别平衡方案

## 1. 问题背景

TASK2 当前是胃镜检查级三标签多标签 MIL 任务。一个样本不是单张图像，而是一次检查目录：

```text
exam_dir = 一个 bag = 多张内镜图像 + 三个检查级标签
```

三个标签为：

- `label_esophageal_smt`
- `label_esophageal_mucosal_or_tumor`
- `label_gastritis`

这三个标签的阳性率相对稳定，当前主要问题不是极端类别稀缺，而是：

- 多图 bag 中关键图像稀释；
- 食管 SMT 与食管黏膜病变/肿物需要更清晰地区分；
- 胃炎可能依赖多区域弥漫性证据；
- 固定阈值可能不是每个标签的最佳阈值；
- 训练集、验证集、测试集必须按患者划分，避免同一患者多次检查泄漏。

## 2. 当前优化目标

当前方案目标是提升三标签分类准确度和泛化稳定性：

1. 保持验证集和测试集真实分布不变。
2. 优先优化 `gastro_label_graph_mil`。
3. 在必要时只对训练集做轻量类别平衡。
4. 系统比较固定阈值和 per-label threshold。
5. 用报告弱监督改善 attention，而不是把诊断文本作为输入。

## 3. 类别平衡原则

三标签任务不建议默认使用强过采样。更推荐以下顺序：

1. 先跑原始采样 + ASL/BCE，建立基线。
2. 如果某个标签 recall 明显偏低，再启用训练集轻量重采样。
3. 重采样单位必须是检查级 bag，不是图片。
4. 验证集和测试集必须保持原始分布。

正确的重采样单位：

```text
一个检查目录 bag
```

错误做法：

```text
把阳性检查中的所有图片复制出来，当作阳性图片级样本
```

原因是检查级阳性不代表该检查目录下每张图都包含目标病变。

## 4. 推荐配置口径

如果启用 `class_balance`，配置中的标签列表必须与三标签任务一致：

```yaml
class_balance:
  enabled: false
  apply_to: train_only
  mode: multilabel_minority_oversample
  target_strategy: per_label_majority
  max_repeat_per_bag: 5
  max_added_records: 2000
  allow_overshoot_ratio: 0.05
  top_candidate_pool: 32
  candidate_sample_size: 256
  prefer_multi_tail_positive: false
  label_names:
    - label_esophageal_smt
    - label_esophageal_mucosal_or_tumor
    - label_gastritis
  tail_labels: []
  report_filename: class_balance_report.json
```

说明：

- `enabled` 默认建议为 `false`，先观察原始采样表现。
- `max_repeat_per_bag` 不宜过大，避免少数检查被反复学习。
- `tail_labels` 在当前三标签任务中默认留空。
- 如后续发现某个标签明显弱，可只把该标签加入重点采样策略。

## 5. 损失函数建议

当前可比较三组：

1. 原始采样 + `asymmetric`
2. 原始采样 + `bce`
3. 轻量重采样 + `asymmetric`

如果三标签分布保持稳定，优先选择训练更稳、验证损失反弹更小的一组，而不是只看训练集 loss。

## 6. 阈值与校准

三标签任务的重点之一是 per-label threshold：

- `label_esophageal_smt` 和 `label_esophageal_mucosal_or_tumor` 可能需要不同阈值来控制互相混淆。
- `label_gastritis` 阳性表现更弥漫，默认阈值可能偏保守或偏宽松。

建议记录：

- 固定阈值 `0.5` 的结果。
- 验证集搜索得到的每标签阈值。
- 使用每标签阈值后的测试集指标。

## 7. 模型优化建议

当前基础模型为：

- `model/gastro_label_graph_mil`

优先优化方向：

1. 调整 `attn_dim`、`dropout`、`feature_dim`。
2. 对标签图传播加入残差门控，防止标签表征过度混合。
3. 加入 attention entropy 或 diversity 正则。
4. 比较 ConvNeXt-Tiny 与 ResNet50 backbone。
5. 比较 `train_max_instances` 和 `eval_max_instances` 对结果的影响。
6. 对三个标签分别查看 top-k attention 图像，定位错误样本。

## 8. 报告弱监督建议

`watch` 和 `specimen` 可以作为训练阶段弱监督：

- `watch` 解析出解剖区域和病变线索。
- `specimen` 解析出活检区域锚点。
- 生成实例相关性软标签或区域先验。

但必须遵守：

- `watchResult` 只用于派生标签。
- `watch` 不作为最终推理输入。
- `specimen` 不作为诊断文本输入，只作为训练辅助目标。

## 9. 实验记录要求

每次 TASK2 训练至少保存并记录：

- `config.yaml`
- `log.csv`
- `test_result.csv`
- `checkpoints/best_macro_f1.ckpt`
- `checkpoints/best_micro_f1.ckpt`
- `checkpoints/best_val_loss.ckpt`

结果解释时必须说明使用哪个 checkpoint 作为测试来源。

推荐输出目录：

```text
outputs/train_runs/task2/
└── <experiment_dir>/
    └── <model_dir>/
        └── <train_dir>/
```

## 10. 当前执行清单

1. 将 TASK2 datalist 重新生成为三标签。
2. 确认 `configs/task2/train.yaml` 中所有标签列表只包含三个标签。
3. 运行 `gastro_label_graph_mil` 作为三标签基线。
4. 对比固定阈值和 per-label threshold。
5. 查看三个标签的错误样本和 attention top-k 图像。
6. 再决定是否启用轻量重采样或报告弱监督。
