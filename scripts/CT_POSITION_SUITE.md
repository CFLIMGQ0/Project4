# CT-RATE切片缺失与位置机制对照（2026-09-19）

## 实验口径

- **50%/75%分类训练**：完整原CT保留首尾后删除相应比例，再在剩余序列中按序号均匀采样训练配置规定的64层。训练、验证、测试采用同一缺失条件；每例固定一个从种子42和原case_index派生的缺失掩码，ACPE与Original PE共用。不是从旧64层缓存删除，也不复制或插值补足。
- 50%保留678例，排除2例；五个原始患者组剩余135/136/136/135/136例。75%保留118例，排除562例；五个原始患者组剩余28/25/19/27/19例。仅过滤原五折归属，不重新划分；测试第k组、验证第k+1组、其余训练。不同缺失条件病例量不同，因此**只能优先比较同一缺失条件下的配对方法**，不能把跨条件变化全部归因于切片缺失。
- 原始索引、物理位置、采集间距和完整原CT长度只写入审计文件，不传给模型。ACPE沿用原模块的等距fallback；Original PE沿用真实的`TimePositionEncoding`（标量MLP位置嵌入），不是`no_pe`。
- 额外`instance_dropout=0`，避免叠加另一次丢弃；冻结训练时相同的ImageNet ConvNeXt-Tiny、三窗预处理与224分辨率。保留同一三标签和目标词掩码临床所见。每折重新训练聚合/分类模块30轮，最低验证损失选择checkpoint，阈值只在验证集选择；另报固定0.5阈值和OOF结果。
- **位置机制分类**：另外使用完整原680例原64层特征、无随机删除、无额外丢弃、隐藏原始坐标。四个新方法各五折，同时重新跑ACPE和Original PE各五折作为完全同协议对照。合计30个分类折；加上缺失实验20折，共50折。
- 分类均值±标准差是五个测试折的样本标准差（ddof=1），不是缺失掩码重复之间的标准差。小样本75%实验不能据小幅差值宣称显著性。

## 四个方法的可复现边界

固定官方仓库提交与所核对文件SHA256见`outputs/ct_rate_680/position_baselines/upstream_manifest.json`。本项目新增实现位于`src/exp_8/position_baselines.py`，不修改原`exp_8/models.py`或ACPE结构；只在新实验构建时关闭ACPE并替换两层切片自注意力的Q/K位置机制，保留原投影、残差、归一化、FFN、MIL、标签图和跨模态融合。

四个新方法配置里的`position_variant=no_pe`只负责禁用原ACPE注入路径，实际位置方法由`ct_attention_variant`决定，并保存在`context_encoder.layers.*.self_attn.position_module.*`权重/缓冲区；它们不是无位置编码基线。统一构建器关闭PyTorch注意力快速路径，防止推理时跳过自定义Q/K机制。

| 方法 | 本实验的明确选择与适配 | 原生标量位置 |
|---|---|---|
| [ComRoPE](https://github.com/Longin-Yu/ComRoPE) | ComRoPE-AP，1轴、块大小2、初始化标准差1；训练反对称生成矩阵，并对Q/K实际施加矩阵指数旋转 | 无 |
| [VideoRoPE](https://github.com/Wiselnn570/VideoRoPE) | 原VideoRoPE，非++；128维头、16/24/24分配、theta=1e6、时间间距2；使用官方低频时间分配和对角坐标公式 | 无 |
| [PaTH](https://github.com/fla-org/flash-linear-attention) | 原因果核与官方数值对齐；方向投影秩32、卷积3+SiLU、L2归一化、2sigmoid门控；双向CT用共享参数的正反核拼接后统一softmax，不启用FoX遗忘门 | 无 |
| [DAPE V2](https://github.com/chuanyang-Zheng/DAPE) | Kerple底座、1×3两层卷积、宽度32、LeakyReLU、原残差；双向CT用绝对相对距离和有效完整注意力图，不沿用LM的未来屏蔽 | 无 |

**VideoRoPE的重要退化**：当前AMEF每张切片只有一个汇总token，即每帧H=W=1，三个对角坐标相同，最终等价于间距2的RoPE。虽然多轴公式已与官方精确对齐，但这个CT实验不能检验原方法完整空间布局和低频分配优势，不能在论文中不加限定地称为完整VideoRoPE复现。PaTH、DAPE V2的双向改动也必须标注为CT适配，不是原论文因果架构。

ComRoPE仓库未提供许可证：仅将公开公式独立实现用于本地研究、将原仓库作为核对参考；对外分发其仓库代码前应确认作者授权。其余上游原有许可证保留在各自参考目录。

## 验证

`audit_ct_position_baselines.py`从固定官方源码抽取纯数学接口进行数值对照：

- ComRoPE的旋转及参数梯度；VideoRoPE多轴Q/K旋转逐元素精确对齐。
- DAPE V2原始因果分数与分数梯度；PaTH官方因果输出，以及独立逐步秩一递推的双精度前向/梯度。
- 自注意力适配器退化为普通注意力时与原模块一致；四个模块均检查padding不影响有效token。DAPE两层卷积之间显式屏蔽无效padding，防止卷积偏置从填充区回流。
- 六模型真实16例数据各两步前向/反向、有限损失/梯度、位置参数确实获得梯度（VideoRoPE无可训练位置参数），并毒化原始坐标验证输出不变。测试成功记录在`implementation_audit.json`和`smoke_audit.json`。

## 位置恢复

`evaluate_ct_position_suite.py`在完整680例分类组全部完成后运行。新ACPE checkpoint仅以五折最低验证损失选择，使用该折原测试患者；所有删除条件0/25/50/75%共用满足75%删除后仍有64层的测试CT。非零条件固定5个种子：42、2026、3407、7919、104729。

复用已有严格位置恢复函数、真实冻结视觉编码器及原生`apro_context_coordinates`，不微调、不使用分类概率。PRE/Acc排除端点，GRE统计全部相邻间隔；每CT先平均随机重复，再对CT等权平均，标准差为CT间样本标准差。提供逐切片结果、采样清单、逐CT/逐重复/总体CSV与曲线。旧原始特征只在患者ID和特征协议一致时复用。

本次另外独立从逐切片CSV重算全部432组指标和CT间标准差，与汇总一致。25%条件等距基线的最大内部误差为0.043386，确实小于0.05，故Acc@0.05=1不是取整或漏算错误。绘图包装接口按每个条件分别选择种子（0%用0，其余用42），修复旧辅助函数只选seed=0而留下三个空白面板的问题；不修改指标或抽样。

**四个新方法没有原生标量c_t，PRE/GRE/Acc标注N/A而不是0或待完成**。脚本实际加载其验证集选出的checkpoint、检查ACPE没有残留，不把角度、矩阵、注意力分数或高维嵌入冒充位置，不新增回归头，不按测试真值拟合。若需给这些方法定义另一种“位置可解码性”任务，须另行制定并报告协议，不能混入本表。

数据空间限制：输入是规则NIfTI体积，只能审计RAS仿射坐标；没有逐层DICOM采集位置，不能从规则仿射矩阵证明原始采集不存在非均匀间距。

## 六GPU队列与磁盘

`ct_six_gpu_queue.py`固定查询204的4张卡和202的2张卡，记录索引、UUID、显存和利用率。默认每10秒更新，按可用显存放任务，**不要求卡完全空闲，也不限制每卡一个任务**。启动前缺卡直接报错；运行中主机失联写入状态和日志，不静默忽略。204的0/1号卡目前被其他任务占满，仅剩几十MiB；它们是可见但暂不可用，不会终止其他人的进程。

特征阶段最多8个任务防止CPU内存和解压争用；这是全局原图解码并发限制，不是每卡任务数限制。202每次只暂存4例完整原CT到`/new_data/Lim/ct_position_suite_20260919/datasets/queue_raw/<job>`，提取并将特征回收204后删除本任务临时副本，不删除原始数据或其他实验。每次新增任务检查至少8GiB磁盘余量，避免整库复制。训练结果逐折从202回收，不等整个主机完成。

任务有独立日志、最多三次尝试、超时、完成标记和严格训练协议核对。**并非保证不会报错**；三次失败会明确列出而非伪报完成。状态超过轮询间隔很久未更新时应检查调度器进程；重新启动前必须核对已有子进程，不能同时重复启动两个调度器。

## 运行与查看

在项目根目录使用：

```bash
PYTHON=/home/Lim/conda/envs/myenv/bin/python
$PYTHON src/scripts/check_ct_position_suite.py
tail -f outputs/ct_rate_680/position_suite/scheduler.log

# 独立六卡可见性检查
$PYTHON src/scripts/ct_six_gpu_queue.py --inventory-only

# 已生成的队列续跑：先确认没有旧调度器或残留任务
$PYTHON -u src/scripts/ct_six_gpu_queue.py

# 单折示例，仅在未被队列调度时手动运行
CUDA_VISIBLE_DEVICES=2 $PYTHON src/scripts/run_ct_position_baselines.py --variant path --fold 1
CUDA_VISIBLE_DEVICES=2 $PYTHON src/scripts/run_ctrate_slice_missing.py \
  --output-dir outputs/ct_rate_680/missing50_raw_hidden_fivefold --variant apro_full --fold 1
```

不要从`src`目录直接启动，避免本项目`statistics.py`遮蔽Python标准库。不要在实验进行时改动被协议固定的模型/训练脚本；失败修复须审计影响并保留旧协议和日志。

主要结果位置：

- `outputs/ct_rate_680/missing50_raw_hidden_fivefold/comparison.md`、`comparison.csv`
- `outputs/ct_rate_680/missing75_raw_hidden_fivefold/comparison.md`、`comparison.csv`
- `outputs/ct_rate_680/position_baselines/comparison.md`、`comparison.csv`
- `outputs/ct_rate_680/position_baselines/position_recovery/all_methods_table.md`、`all_methods_summary.csv`
- `outputs/ct_rate_680/position_suite/status.json`、`history.json`、`logs/`：真实队列状态、主机/GPU分配及逐次日志
