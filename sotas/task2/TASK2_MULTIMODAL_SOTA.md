# TASK2 图文多模态 SOTA 适配实验

## 实验边界

本实验用于补充论文表2中的图像与文本联合输入对照。所有模型读取相同的检查级图像 bag 和类别名称掩码后的 `watch` 字段，不读取标签来源字段 `watchResult`。数据划分、类别平衡、图像数量、优化器和训练轮数均与表2主实验保持一致。

这些方法原本面向胸片、皮肤图像或生物医学文献图像，无法在胃镜检查级数据上逐行照搬。因此，项目复现的是各论文的核心融合机制，并将单图视觉 token 改为一次检查中的图像序列 token。表格中统一使用“任务适配复现”标记，不能写成原作者在本数据集上的官方结果。

## 纳入方法

| 项目模型名 | 论文方法 | 年份 | 保留的核心机制 |
|---|---|---:|---|
| `task2_hasan_itf_2024` | Image and Text Feature Based Multimodal Learning | 2024 | 图像与文本特征拼接、轻量 CNN 分类器 |
| `task2_mmfnet_2024` | MMFNet | 2024 | 移位窗口视觉注意力、异构双向交叉注意力、分支辅助监督 |
| `task2_saif_2025` | SAIF | 2025 | 图像到文本映射、相关性掩码、对称 KL 对齐、无参数伪注意力 |
| `task2_mmtf_2025` | MMTF | 2025 | 图文多尺度 token、低注意力 token 跨模态子空间交换 |
| `task2_radfuse_2025` | RadFuse | 2025 | 模态内自注意力、双向交叉注意力、批内 InfoNCE |

论文来源：

- Hasan 等，DOI：<https://doi.org/10.5220/0012438400003657>。
- MMFNet，DOI：<https://doi.org/10.1109/BIBM62325.2024.10822724>。
- SAIF，DOI：<https://doi.org/10.1016/j.patcog.2025.111715>；官方代码：<https://github.com/busitl/Self-Adaptive-Image-Text-Fusion>。
- MMTF，DOI：<https://doi.org/10.1016/j.bspc.2025.108318>；官方代码：<https://github.com/GUESSZERO4/MMTF>。
- RadFuse，DOI：<https://doi.org/10.1016/j.imu.2025.101672>。

## 公平性设置

- 图像编码底座：`convnext_tiny`。
- 文本编码器：TextCNN，词表大小8192，卷积核为2、3、4。
- 图像输入：每次检查最多64张，保持统一采样策略。
- 输入文本：类别名称掩码后的 `watch`。
- batch size：1；梯度累积：4。
- 训练：30轮，AdamW，学习率0.00002，权重衰减0.02。
- 结果：读取验证集 Macro F1 最优 checkpoint 对固定内部测试集的评估。

RadFuse 原始批内 InfoNCE 在局部 batch size 为1时没有负样本。本项目保持 batch size 不变，增加容量为256的跨 batch 特征队列提供历史负样本，使对比目标仍能参与训练；该项属于适配实现的一部分。

## 运行与汇总

GPU恢复正常后执行：

```bash
bash scripts/run_t2_multimodal_sotas_tmux.sh
```

查看进度：

```bash
tmux list-windows -t t2_multimodal_sotas
tmux capture-pane -p -t t2_multimodal_sotas:gpu0 -S -30
```

五个结果全部完成后更新 `table.md`：

```bash
python scripts/task2_multimodal_sotas_table.py --update-table
```
