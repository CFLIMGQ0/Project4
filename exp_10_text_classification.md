# exp10_text_classification：五种文本编码器三标签分类

## 1. 实验目标

`exp10_text_classification` 是纯文本消融实验。模型只读取医生书写的内镜所见字段 `watch`，先遮蔽能够直接暴露答案的类别词，再使用五种文本编码器提取报告表征，最后统一接入 MLP 输出三个独立 sigmoid 标签。

本实验是三标签多标签分类，不是三选一分类。三个标签为：

- `label_esophageal_smt`
- `label_esophageal_mucosal_or_tumor`
- `label_gastritis`

## 2. 答案词遮蔽

`watchResult` 是生成三标签的诊断结论字段，整个 exp10 禁止把它作为模型输入。正式文本固定为 `watch`。

在 tokenize 之前，训练、验证和测试统一遮蔽以下类型的直接答案词：

- 食管 SMT、SMT、食管黏膜下肿物、黏膜下肿瘤等；
- 食管黏膜病变、食管肿物、食管占位、食管新生物、SESCC 等；
- 慢性胃炎、萎缩性胃炎、糜烂性胃炎、胆汁反流性胃炎等。

遮蔽词表直接引用 `tasks/task2/selection.py` 中实际用于生成标签的 `TASK2_LABEL_RULES`。胃炎分级代码还会覆盖 `C1/C2/C3/O1/O2/O3` 的连字符、空格和全角写法，例如 `C-1`、`C 1`、`Ｏ－１`。正则使用字母数字边界，不会误遮蔽 `20cm`、`c10` 等无关内容。

所有答案词统一替换为 `MASKTARGET`。模型仍可以看到食管、胃窦、贲门、隆起、表面光滑、充血、糜烂、萎缩界线等非答案描述。

脚本会生成 `mask_audit.json`，记录各划分的遮蔽数量、命中词频、残留答案词数量和患者交叉情况。只要发现遮蔽后仍残留答案词或患者划分交叉，训练会直接终止。

## 3. 五种文本编码器

五个模型使用相同的数据划分、最大文本长度、损失函数、阈值搜索和 MLP 分类头，只改变文本编码器。

| 模型名 | 编码方式 | 目的 |
|---|---|---|
| `hashed_mean_encoder` | 论文 HashTextEncoder 风格的稳定哈希 ID、token projection、masked mean pooling | 复现当前论文轻量哈希文本分支的纯文本版本 |
| `vocab_attention_encoder` | 训练集字符词表 embedding + token attention pooling | 比较显式词表和可学习关键词权重是否优于哈希均值池化 |
| `textcnn_encoder` | 训练集字符词表 embedding + 2/3/4 尺度 TextCNN | 捕捉局部部位—形态短语 |
| `bigru_encoder` | 训练集字符词表 embedding + 双向 GRU | 建模医生报告的前后文和描述顺序 |
| `transformer_encoder` | 训练集字符词表 embedding + 位置编码 + 两层 Transformer | 建模较长距离的部位—病灶关系 |

非哈希模型的词表只根据训练集遮蔽文本构建，验证集和测试集未登录 token 统一映射为 `<unk>`。

## 4. 统一 MLP 分类头

每种编码器输出 512 维报告表征，末端统一连接：

```text
text representation [512]
  -> LayerNorm
  -> Linear(512, 256)
  -> GELU
  -> Dropout(0.2)
  -> Linear(256, 3)
  -> three sigmoid probabilities
```

训练使用带训练集 `pos_weight` 的 `BCEWithLogitsLoss`。验证集逐标签选择 F1 最优阈值，阈值锁定后在测试集评估。

## 5. 数据划分与指标

- 随机种子：2026；
- 患者级划分：训练/验证/测试为 6:2:2；
- 最大文本长度：128；
- 主指标：Macro F1；
- 其他指标：Micro F1、Macro ROC-AUC、Macro PR-AUC、Subset Accuracy、Hamming Loss、Kappa 和三个标签的独立指标。

## 6. 运行方式

先检查遮蔽、患者划分和训练集词表：

```bash
python -m exp_10.train_text_classification --audit-only
```

依次训练五种编码器：

```bash
python -m exp_10.train_text_classification
```

只训练指定编码器：

```bash
python -m exp_10.train_text_classification --models hashed_mean_encoder,textcnn_encoder
```

所有结果保存到：

```text
/home/Lim/Project4/outputs/train_runs/task2/exp10_text_classification/
```

每个编码器目录包含 `best_model.pt`、`history.json`、`test_metrics.json` 和 `test_predictions.csv`，根目录包含 `mask_audit.json`、`vocabulary.json` 与五模型汇总 `summary.json`。
