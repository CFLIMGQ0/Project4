# CT-RATE固定680例下载

## 范围与来源

每名患者保留一次已经人工标注的检查、重建1，共680个三维CT压缩文件。一个`.nii.gz`文件包含连续多张切片，不是单张二维图像。

- 官方仓库：<https://huggingface.co/datasets/ibrahimhamamci/CT-RATE>
- 固定版本：`deeca4d89e9f978d4d1bccd88a55071ddbb146bb`
- 固定名单：`outputs/dataset_candidates/ct_rate_680_file_sizes.json`
- 影像总大小：127,785,411,565字节，127.79 GB，119.01 GiB。
- 标签来源：作者[公开人工标注报告子集](https://github.com/ibrahimethemhamamci/CT-CLIP/tree/main/text_classifier/data)，按原始映射连接到CT-RATE后按患者去重；不是全量自动提取标签。
- 阳性数：肺气肿148例、肺不张169例、肺纤维化后遗改变188例。
- 三标签均阴性305例、单阳性259例、恰好两项阳性102例、三阳性14例；至少两项阳性116例。

此处下载的是完整检查体积，尚未进行切片抽样、特征提取或训练。检查级标签不提供逐层病灶标注，异常覆盖的连续切片数仍需影像核查。

## 报告来源与图文对应

报告由伊斯坦布尔Medipol大学医院放射科医生在临床工作中撰写，土耳其语原文经机器翻译及双语高年级医学生校对后公开英文版本。参见[原论文方法4.2节](https://arxiv.org/html/2403.17834v1#S4.SS2)。这些报告不是本项目用视觉模型生成的文本。

下载官方`train_reports.csv`及`validation_reports.csv`，使用精确的`VolumeName`逐一连接影像。保留临床信息、检查技术、影像所见和诊断印象四个原始字段，不修改文本。三标签单独存储，下载脚本不会把标签拼入报告。

原始报告包含诊断印象，人工标签也来自报告标注。后续多模态实验需另行固定所见段选取及掩码方案，不能将本次原文保存当作已经完成的信息泄露处理。

## 本地文件

相对项目根目录`/xmlg/Lim/Project4`：

- `datasets/ct_rate_680/dataset/train_fixed/`、`valid_fixed/`：选定680例影像。
- `datasets/ct_rate_680/samples_680.csv`：680例影像路径、标签、对应原始报告各段。
- `datasets/ct_rate_680/metadata_680.csv`：680例采集元数据。
- `datasets/ct_rate_680/selection_manifest.json`：固定选择名单。
- `datasets/ct_rate_680/dataset/radiology_text_reports/`、`metadata/`：用于筛选及核查的官方源表。
- `outputs/ct_rate_680/download/status.json`：下载状态、完成例数、已校验字节数。
- `outputs/ct_rate_680/download/verified_files.json`：逐例SHA256、NIfTI形状、间距与方向。
- `outputs/ct_rate_680/download/failed_files.json`：下载失败清单。
- `outputs/ct_rate_680/download/table_verification.json`：图文标签配对核查。

临床数据仅在本地使用，不随源码推送公开仓库。HF令牌使用HF凭据存储，不写入脚本、实验表格或日志。

## 运行和检查

先完成官方访问授权并在服务器认证，再从`src`运行：

```bash
python -u scripts/download_ctrate_680_subset.py --workers 4 --network-mode v2raya
```

支持HTTP断点续传，完整文件会复用；重跑时重新核对SHA256。脚本通过文件锁阻止重复并发任务，不解压整个数据集，不下载其他重建版本。

2026-09-14按用户要求改为通过本机V2rayA继续下载，HTTP代理为`http://127.0.0.1:21171`，启动时选用已配置的“日本高速09”节点。此前已下载并校验54例，约9.65 GB，继续复用完整文件和已有断点。

`--network-mode v2raya`会清除下载进程继承的代理设置，再显式指定V2rayA的HTTP代理；HF会话使用`trust_env=False`，代理故障时不自动切换至Mihomo或直连。不修改系统代理、VPN服务或其他项目的网络设置。需要直连时使用`--network-mode direct`（省略参数也默认直连）。

```bash
tail -n 10 /xmlg/Lim/Project4/outputs/ct_rate_680/download/download_v2raya.log
```

本次后台会话为`ctrate680_v2raya`，采用4路并发，关闭当前终端后继续下载。`download.log`保留上一次下载记录，本次追加写入`download_v2raya.log`。重启任务后会重新校验已有文件，因此进度计数会从本轮已校验数开始增长，磁盘上的完整文件仍然保留。

只有`status.json`中的`status`为`complete`、`verified`为680且`failed`为0，才代表全量下载完成。

## 校验与预处理注意

逐例核对官方字节数、LFS SHA256及三维NIfTI头，并保存形状、间距与方向。已选名单与官方`no_chest_train.txt`、`no_chest_valid.txt`按影像文件名核对，交集为0。后续仍应检查图像质量及阳性异常的连续层面分布。

采用官方`train_fixed`和`valid_fixed`，原始强度和间距修正说明保存在`dataset/data_correction_note.md`。后续预处理应先核对修正后的NIfTI头，避免重复应用原始DICOM强度变换。
