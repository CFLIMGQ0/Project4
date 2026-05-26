# TASK1 SOTA 说明

## 目的

`sotas/` 目录用于放置比基础 MIL baseline 更强的代表性方法。这里的实现不是机械复刻原论文代码，而是基于经典 WSI / MIL SOTA 思路，按当前项目的胃镜多标签 bag-level 训练接口做适配，保证：

- 输入输出格式与现有训练流程一致
- 自动运行方式与 baseline 保持一致
- 训练参数默认继承 `configs/task1/train.yaml`
- 训练结束后立即测试

## 目录结构

- SOTA 模型目录：`sotas/task1/gastro_sota/`
- 自动运行配置：`configs/task1/auto_sotas.yaml`
- 自动输出目录：`/home/Lim/Project4/outputs/train_runs/task1/sotas`

## 当前 5 个胃镜 SOTA 模型

1. `gastro_clam_sb_sota`
   说明：基于 CLAM 的单分支版本，强调 bag 级注意力与关键实例约束。
   参考：CLAM, Nature Biomedical Engineering 2021
   链接：`https://www.nature.com/articles/s41551-020-00682-w`

2. `gastro_clam_mb_sota`
   说明：基于 CLAM 的多分支版本，为不同标签学习更细粒度的注意力与实例聚类约束。
   参考：CLAM, Nature Biomedical Engineering 2021
   链接：`https://www.nature.com/articles/s41551-020-00682-w`

3. `gastro_dsmil_sota`
   说明：基于 DSMIL 的关键实例 + bag 双流聚合思路，适合显著病灶驱动场景。
   参考：DSMIL, CVPR 2021
   链接：`https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Dual-Stream_Multiple_Instance_Learning_Network_for_Whole_Slide_Image_Classification_CVPR_2021_paper.pdf`

4. `gastro_transmil_sota`
   说明：基于 TransMIL 的相关性建模思路，用 Transformer 编码实例间关系。
   参考：TransMIL, NeurIPS Workshop / arXiv 2021
   链接：`https://arxiv.org/abs/2106.00908`

5. `gastro_dtfd_mil_sota`
   说明：基于 DTFD-MIL 的双层伪 bag 蒸馏思路，增强大 bag 的层级聚合能力。
   参考：DTFD-MIL, CVPR 2022
   链接：`https://arxiv.org/abs/2203.12081`

## 自动运行规则

- 在 `configs/task1/train.yaml` 中将 `auto_sotas: true` 打开后，会读取 `configs/task1/auto_sotas.yaml`。
- 训练超参数默认继承 `configs/task1/train.yaml`。
- `auto_sotas.yaml` 负责补充模型结构参数与各模型专有超参数。
- 输出目录固定编号：
  - `train_001_gastro_clam_sb_sota`
  - `train_002_gastro_clam_mb_sota`
  - `train_003_gastro_dsmil_sota`
  - `train_004_gastro_transmil_sota`
  - `train_005_gastro_dtfd_mil_sota`

## 断点续跑规则

- 已存在完整测试结果的模型会自动跳过。
- 存在 `checkpoints/last.ckpt` 的未完成训练会自动续跑。
- 没有断点但目录有残留时，会清理该目录并重新训练。

## 输出文件

- `remark.txt`：整体摘要
- `notes.json`：结构化对比结果
- `log.csv`：epoch 级训练日志
- `test_result.csv`：3 个最佳 checkpoint 的测试结果
- `checkpoints/last.ckpt`：断点续训入口

## 运行方式

更推荐直接在 `configs/task1/train.yaml` 中打开 `auto_sotas: true`，然后执行：

```bash
python train.py
```

前提：

- `configs/task1/train.yaml` 中设置 `auto_baselines: false`
- `configs/task1/train.yaml` 中设置 `auto_sotas: true`
