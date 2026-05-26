# TASK1 消融实验说明

本消融只针对 TASK1 胃镜 examination-level multi-label classification，不影响 TASK2。

## 实验分组

输出根目录统一为：

```text
outputs/train_runs/task1/task1_ablation/
```

当前 `configs/task1/auto_ablations.yaml` 只启用 1 个实验组：

1. `exp_task1_ablation1`
   - `exp_task1_ablation1_full_label_graph`
   - `exp_task1_ablation1_wo_label_graph`
   - `exp_task1_ablation1_label_self_attention`
   - `exp_task1_ablation1_static_gcn`
   - `exp_task1_ablation1_dynamic_gat`
   - `exp_task1_ablation1_label_transformer`
   - `exp_task1_ablation1_low_rank_graph`
   - `exp_task1_ablation1_cosine_graph`
   - `exp_task1_ablation1_label_mlp_mixer`
   - `exp_task1_ablation1_label_hypergraph`

本轮只验证 `label graph reasoner` 的作用，不做 backbone 公平性对比、不做多随机种子稳定性实验，也不做 label-wise attention / shared attention 消融。

当前一键运行启用了 TASK1 专用自动探索流程：

1. 固定 `train_max_instances=16`，先跑 10 个模块结构消融。
2. 选出阶段 1 中 `macro_f1` 最高的模块。
3. 固定该模块，跑 `train_max_instances=8/12/16/20/24`。
4. 如果最佳 `train_max_instances` 不是 `16`，则在最佳实例数下重新跑 10 个模块结构消融。
5. 使用最终最佳模块与最佳 `train_max_instances`，固定 `ConvNeXt-Tiny` backbone，运行最终 11 个模型对比：`Label graph MIL`、`Attention MIL`、`Mean pooling`、`Transformer-context MIL`、`Top-k MIL`、`Max pooling`、`TransMIL`、`DSMIL`、`DTFD-MIL`、`CLAM-MB`、`CLAM-SB`。

该流程的探索实验数为 15 个；如果触发第 4 步，则探索实验数为 25 个。最终模型对比会额外运行 11 个固定模型。

## 模块开关

主模型 `GastroLabelGraphMIL` 支持以下配置字段：

```yaml
use_label_graph: true
label_graph_type: learnable    # learnable / self_attention / static_gcn / dynamic_gat / label_transformer / low_rank_graph / cosine_graph / label_mlp_mixer / label_hypergraph
use_label_wise_attention: true
attention_type: label_specific  # label_specific / shared / none
pooling_type: label_attention   # label_attention / shared_attention / mean
```

本轮 10 个配置保持相同的 ConvNeXt-Tiny encoder、label-wise gated attention pooling、分类头、训练策略、数据划分和随机种子控制，只改变 label graph reasoner：

| 实验 | 修改内容 | 目的 |
|---|---|---|
| `exp_task1_ablation1_full_label_graph` | 使用原始可学习 label-token 图传播 | 作为完整模型对照 |
| `exp_task1_ablation1_wo_label_graph` | 移除 label graph reasoner，attention 输出直接进分类头 | 验证标签图关系建模是否有效 |
| `exp_task1_ablation1_label_self_attention` | 用通用 label self-attention reasoner 替换原始图传播 | 对比 Transformer/self-attention 式标签关系建模 |
| `exp_task1_ablation1_static_gcn` | 用训练集标签共现先验构建静态图，再做 GCN 式传播 | 对比 ML-GCN 类静态共现图方法 |
| `exp_task1_ablation1_dynamic_gat` | 根据当前样本的 label embeddings 动态估计标签注意力图 | 对比 GAT/动态图类标签关系建模 |
| `exp_task1_ablation1_label_transformer` | 用一层 Transformer encoder 建模标签上下文 | 对比标准 Transformer 标签关系建模 |
| `exp_task1_ablation1_low_rank_graph` | 用低秩可学习邻接矩阵做标签传播 | 对比参数更少的可学习标签图 |
| `exp_task1_ablation1_cosine_graph` | 根据 label embeddings 的余弦相似度动态构图 | 对比无显式邻接参数的相似度图 |
| `exp_task1_ablation1_label_mlp_mixer` | 用 MLP-Mixer 风格 token mixing 混合标签表征 | 对比不使用 attention/GCN 的标签混合结构 |
| `exp_task1_ablation1_label_hypergraph` | 用可学习超边聚合多标签高阶关系 | 对比 Hypergraph 类高阶标签关系建模 |

## 运行方式

一键运行（默认即 TASK1 消融）：

```bash
python train.py
```

显式运行（与一键等价）：

```bash
python train.py --task task1 --auto-ablations-config configs/task1/auto_ablations.yaml
```

如果要切回 TASK2，需要显式指定：

```bash
python train.py --task task2
```

训练完成后会自动尝试生成汇总文件；也可以手动重新汇总：

```bash
python scripts/task1_ablation_summary.py
```

汇总文件默认写入：

```text
outputs/train_runs/task1/task1_ablation/results/
├── ablation_summary.csv
└── ablation_summary.md
```

自动探索流程的汇总文件写入：

```text
outputs/train_runs/task1/task1_ablation/exp_task1_auto_module_instance_search/
├── module_instance_search_summary.csv
├── module_instance_search_summary.json
├── module_instance_search_summary.md
└── final_models_best_params/
```

每个训练目录仍按原 pipeline 保存 `config.yaml`、`log.csv`、`test_result.csv`、`test_report.csv`、`checkpoints/` 和各 checkpoint 的测试产物。
