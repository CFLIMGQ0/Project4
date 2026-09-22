# CT-RATE切片缺失后重新训练的配对对照

## 本次口径

- 用户最终选择删除25%，不是75%，也不是从已有64层缓存删除25%。
- 从完整RAS标准化CT中随机删除25%的原始切片，保留完整首尾；在剩余序列中按序号均匀采样训练配置指定的T张。本项目T=64。
- 剩余不足T的患者在两个模型及所有角色中统一排除，不重复、不插值。本次680例全部符合条件，排除0例。
- 只过滤原患者五折，不重新划分。测试第k折，验证下一折，其余训练。本次仍为408/136/136。
- 缺失种子42，每例以`SeedSequence([42, source_case_index])`派生独立固定种子；只做一次缺失抽样。训练、验证、测试和两个模型共用同一病例的同一组切片，不逐轮重新抽样。
- 原位置恢复实验的抽样函数直接复用；不使用其测试预测或分类checkpoint筛选本次病例、随机种子或超参数。
- 复用固定ConvNeXt-Tiny编码器、RAS方向、三窗、224像素预处理，只重新提取所选64层的特征。原图像和既有缓存不修改。
- 模型只输入视觉特征、掩码和原临床目标词掩码文本；`instance_indices=None`、`original_image_counts=None`。ACPE使用现有等距fallback，Original PE使用现有`TimePositionEncoding`。真实原索引、长度和位置只存在于采样审计文件和缓存中。
- 两模型分别使用已有`apro_full`、`original_pe`变体，不修改模型结构。冻结视觉编码器，其余模块从头训练；不加载旧分类checkpoint。
- 保留30轮、ASL、0.01标签查询辅助损失、相同优化器、每折种子与阈值选择。额外`instance_dropout`设为0，避免25%缺失后又随机丢片。
- 主指标来自每折验证集选阈值后的测试Macro-F1；另存固定0.5阈值结果。均值±五个测试折的样本标准差，`ddof=1`，不是多次缺失随机抽样的标准差。
- 这是图文多模态分类对照，不是纯视觉位置回归；不能单凭分类分数证明坐标恢复有效。与旧实验的位置输入、训练额外切片丢弃不同，不应把对旧实验的分数差全部解释为ACPE作用。

## 运行

在项目根目录、已有`myenv`环境中：

```bash
python -u src/scripts/run_ctrate_slice_missing_pool.py --delete-fraction 0.25 --feature-workers 8
```

流水线依次进行抽样、特征提取、特征/协议审计、两模型前向反向和原坐标泄漏检查、202主机同步及协议审计、两模型五折、汇总。

202主机通过既有SSH密钥连接。原始CT在本机，因此特征提取在本机有显存的卡运行；只将共用的小型特征缓存同步到202进行训练，不向空间紧张的远程盘复制完整原始数据集。

调度默认每15秒检查显存，同卡允许多个任务，新进程启动后暂时预留显存，避免同时启动时读取到尚未分配的旧空闲值。默认门槛3500MiB，可用`--minimum-free-mib`调整。启动和同步前检查至少3GiB磁盘空间；不清理其他实验数据。失败任务最多尝试3次，每次保留单独日志。

重启同一命令可跳过已完成特征及与协议匹配的已完成训练。已完成结果不覆盖；协议不同则明确报错。不要并行启动两个相同输出目录的流水线。

输出：`outputs/ct_rate_680/missing25_raw_hidden_fivefold/`。

- `data/sampling.json`：完整删除索引、保留/采样索引、原始病例ID、每例种子、实际删除比例、排除名单。
- `data/samples.json`和`data/patient_folds.json`：保留病例与原五折对应关系。
- `data/features/`与`data/feature_integrity.json`：共用特征及逐文件哈希。
- `data_audit.json`、`smoke_audit.json`：输入核对、有限损失/梯度、原始位置元数据篡改不影响输出的检查。
- `preprocessing_overlap_audit.json`：本次人工运行的预处理回归核对；32例615张重叠切片与原训练缓存的视觉特征完全一致。
- `apro_full/`、`original_pe/`：独立协议、每折训练历史、checkpoint、测试预测、五折汇总。
- `comparison.csv`、`comparison.json`、`comparison.md`：两模型对照。
- `gpu_pool_status.json`、`logs/`、各阶段`job_history.json`：进度、显卡分配、每次任务结果。

独立核对及汇总：

```bash
python src/scripts/run_ctrate_slice_missing.py --audit-only
CUDA_VISIBLE_DEVICES=2 python src/scripts/run_ctrate_slice_missing.py --smoke-test
python src/scripts/run_ctrate_slice_missing.py --aggregate
```

更换缺失比例或种子时必须使用新的`--output-dir`，不可混入本次输出。

## 本次执行中修复的问题

首次完整训练在记录第1轮损失时发现项目`src/statistics.py`遮蔽了Python标准库，前向反向本身通过但`statistics.mean`不存在。入口已去掉过早添加`src`到搜索路径的操作，复用原训练入口的导入顺序，并在冒烟检查中验证标准库的`mean/stdev`。未修改项目同名文件或模型结构。失败的训练协议、部分文件及日志归档到`failed_import_shadowing/`，没有完成的测试结果参与汇总；680例特征缓存不受影响。

本次CT十个任务已在后续MR四标签修复前完成，使用的源码保存在结果目录`source_snapshot/`。后续共用文件的源码哈希发生变化，旧CT结果协议不重新标记；复现实验应使用该快照或在新输出目录运行当前源码。MR修复已验证三标签logits和辅助损失逐位不变。
