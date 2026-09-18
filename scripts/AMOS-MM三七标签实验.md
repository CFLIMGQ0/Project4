# AMOS-MM 三标签与七标签共存实验

## 任务与样本

三标签：肝脏、肾脏、胆囊异常。
七标签：上述三项加脾脏、肠道、胰腺、胃异常。
这些是检查级器官异常标签，不是器官分割标签或七种单一疾病。

两个任务原始清单共用 1,687 个已配对检查：1,487 例自动标注用于开发，200 例人工标注始终独立测试。
官方 AMOS-MM 压缩包内 `amos_5964` 影像本身是截断 gzip，无法恢复；同编号 AMOS22 Zenodo 影像尺寸与内容均不一致，不能替换。
因此纯文本仍使用全部 1,487 例开发报告，纯图像和多模态仅从训练/验证开发集排除该 1 例、使用 1,486 例；固定 200 例人工测试集完全不受影响。
开发数据固定分成五折，每次四折训练、一折验证，各模型训练五次。
这是“五折开发、固定人工测试集”，不是五个互斥测试折的 OOF 实验。
尚无可核实的跨检查患者标识，因此划分单位准确称为检查 ID。

自动标注中的 `uncertain`、`not_stated` 按标签屏蔽，不能作为阴性监督。
掩码同时应用于主分类、CLAM 实例、DTFD 伪包、MMFNet 图文分支与 AMEF 查询辅助损失。
三标签完全无已知目标的开发检查有 12 例，七标签有 7 例；这些检查不产生分类监督，仍保留在统一检查划分中。
固定人工测试集的标签完整；三标签共阳性有 84 例，七标签共阳性有 121 例。

## 输入和模型

图像从 RAS 标准化 CT 的完整轴位序列均匀选取最多 64 层；不足 64 层时全部使用。
三窗（窗位/窗宽）固定为 40/400、60/150、40/800，经 224×224 缩放与 ImageNet 归一化。
冻结 ImageNet ConvNeXt-Tiny 的 768 维特征仅提取一次，两种任务、所有图像方法共用；损坏病例不生成也不伪造特征。
APro-CoPE 保留原始层索引与原序列长度，训练丢弃部分实例后仍使用原位置。

文本仅拼接官方报告的胸部、腹部、盆腔 `findings`，长度上限 512 token。
不输入 `impression`、问答、分类标签或扫描 ID；原报告文件保持不变。
固定诊断词典适用于全部检查和两种任务，使用 `MASKTARGET` 替换直接诊断词，保留形态、密度、位置等所见。
标签本身来自同份报告，因此该实验是报告辅助分类；词典掩码不代表已经完成独立标签泄漏审计。
词表仅使用当前训练折文本构建；哈希编码使用项目既有确定性规则。

每种任务有 20 个模型，沿用项目中表 2 的任务适配实现：

- 10 个纯图像：Attention MIL、Mean pooling、Transformer-context MIL、Top-k MIL、Max pooling、TransMIL、DSMIL、DTFD-MIL、CLAM-MB、CLAM-SB。
- 5 个纯文本：Hashed mean、Vocabulary attention、TextCNN、BiGRU、Transformer。
- 4 个多模态基线：MMFNet、RadFuse、SAIF、MMTF。
- AMEF-MIL 的完整图文融合预测路径。

合计 2 × 20 × 5 = 200 个正式训练任务。
AMEF 保留现有融合结构，查询判别项推广到 C 标签的 C−1 个循环置换，权重 0.01。
三标签时仍是原来的两个循环置换；只有两端监督都明确的正负标签对参与该损失。
本次不额外训练纯图像蒸馏分支。

## 训练与评估

图像/多模态训练 30 轮，batch size 16，AdamW 学习率 0.0002，weight decay 0.02，20% 预热后余弦衰减。
主损失为已知标签 ASL，叠加各模型原权重的辅助项；按验证 ASL 最小选权重。
纯文本最多 50 轮，batch size 64，按训练折已知正负数设置 BCE 权重，验证 macro-F1 连续 8 轮未提升早停。
图像/多模态基线采用 AMP；文本和 AMEF 采用 FP32。梯度裁剪范数上限 5。

阈值只在开发验证集的已知标签上从 0.10–0.90（间隔 0.05）选取；某标签验证集缺少一类时固定为 0.5。
先保存最优权重和验证阈值，再评估一次固定人工测试集。
报告五次测试 macro-F1 的均值与样本标准差，另存固定 0.5 阈值 F1、逐标签 F1、micro-F1、AUROC、共阳性子集 F1 和完全匹配率。
五次评估对象是相同 200 例，不能把预测拼成 1,000 个独立样本。

## 文件与启动

源码新增于 `src/scripts/`，不改旧实验代码、结果或论文。
输入协议、检查清单、五折与特征位于 `outputs/amos_mm/experiment/`。
训练协议、任务清单、每折 checkpoint/预测/日志和汇总位于 `outputs/amos_mm/all_models/`。
`summary.json` 与 `summary.csv` 随任务完成更新，`completed_runs < 5` 仅表示中间进度。
模型错误单独写入各任务的 `error.json`，队列继续其他任务，汇总不会将失败任务算作完成。
工作进程使用文件锁，防止重复运行同一任务；协议 SHA256 防止不同代码与输入混算。

在 `src` 下使用现有 `myenv` 环境运行：

```bash
python scripts/prepare_amos_mm_experiments.py
CUDA_VISIBLE_DEVICES=1 python scripts/prepare_amos_mm_experiments.py --features
CUDA_VISIBLE_DEVICES=1 python scripts/run_amos_mm_all_models.py --smoke
python scripts/run_amos_mm_all_models.py --initialize
CUDA_VISIBLE_DEVICES=1 python scripts/run_amos_mm_all_models.py --worker 0
CUDA_VISIBLE_DEVICES=1 python scripts/run_amos_mm_all_models.py --worker 1
python scripts/amos_mm_remote_pool.py --stage
python scripts/run_amos_mm_gpu_pool.py
```

动态 GPU 池管理本机 4 张卡和远端 2 张卡，显存空闲不少于 12,000 MiB 时才启动本机槽位；每张卡最多运行一个本实验任务。
远端每卡一个长驻调度槽，本机被其他用户占用的卡会等待，释放后自动补上任务。
每个训练进程最多使用单卡 40% 的 PyTorch 分配显存，不终止其他项目的 GPU 任务。
`smoke_tests.json` 仅记录实现检查，合成输入上的数值不用于实验性能报告。
全部 200 个任务成功后，调度器自动把最终汇总写入已清空的根目录 `temp.md`；中途不会把不完整结果写入该文件。
`monitor_amos_mm_unattended.py` 每 60 秒校验结果文件和预测数组，单任务失败最多自动尝试 3 次，本机 30 分钟无 epoch 心跳或远端单任务超过 6 小时时自动重启对应槽位。
监控同时检查本机 `/xmlg`、系统盘和远端系统盘、`/new_data`；若项目盘空闲低于 30 GiB，会先校验 SHA256，再把已完成 checkpoint 分散转存到两块远端磁盘，保留指针记录。
`run_amos_mm_watchdog.sh` 在监控进程意外退出而任务尚未结束时等待 30 秒自动拉起，避免单次监控异常导致后续无人巡检。
