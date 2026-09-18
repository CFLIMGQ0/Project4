# AMOS-MM 数据下载

## 下载范围与目录

下载官方公开训练、验证 CT 以及原始报告，供后续外部数据集实验使用。本步骤不启动训练，不修改报告文本或论文。

来源：[AMOS 官方数据页](https://era-ai-biomed.github.io/amos/dataset.html)；其中 AMOS-MM 的下载入口为 [Google Drive 文件夹](https://drive.google.com/drive/folders/1R4DaN0dT8ef_0l48bHEuswG1p-MsYclX)。下载进程显式使用本机 V2rayA 的 HTTP 代理 `127.0.0.1:21171`，不修改系统代理配置。

| 内容 | 数量 | 实际存放位置 |
| --- | ---: | --- |
| 官方训练 CT | 1,287 | `/home/Lim/datasets/amos_mm/imagesTr`，本机系统盘 |
| 官方验证 CT | 400 | `/xmlg/Lim/Project4/datasets/amos_mm/imagesVa`，本机项目盘 |
| 原始报告 | 1,687 个检查条目 | `/xmlg/Lim/Project4/datasets/amos_mm/report_generation_train_val.json` |
| 官方 VQA 元数据 | 原文件 | `/xmlg/Lim/Project4/datasets/amos_mm/vqa_train_val.json` |

项目内的 `datasets/amos_mm/imagesTr` 为指向实际训练存储目录的软链接；图像读取入口仍统一为 `/xmlg/Lim/Project4/datasets/amos_mm`。两块物理盘各保留至少 20 GiB 空闲空间。

## 官方目录核对

训练 ZIP 含 1,290 个 CT 文件，验证 ZIP 含 400 个 CT 文件。`amos_7008`、`amos_7312`、`amos_7333` 在两个包中重复出现，大小和 ZIP CRC32 一致。官方报告将这三例列入验证集，因此仅下载报告指定的验证文件，跳过对应的三份训练副本。实际下载 1,687 个检查图像，约 73.0 GiB；不额外保存整个 ZIP，也不解压 `.nii.gz` 内部的体积数据。

这 1,687 个检查均能匹配官方报告；先前 LEAVS 审计生成的 1,211 例三标签候选清单也全部匹配。候选清单为本项目的标签完整性筛选结果，并非官方划定的数据子集。检查编号匹配不代表已经核实患者唯一性；后续划分实验时仍需遵循预定的标注来源和独立测试集安排。

下载保留原始报告文件。官方 JSON 的 `licence` 字段为 `CC BY-ND 4.0`，来源与声明一并记录于 `data_mapping.json`。

## 完整性与续传

先通过 HTTP Range 读取 ZIP 中央目录，再逐文件流式解包。每个文件同时核对官方 ZIP 声明的字节数和 CRC32，计算本地 SHA256，并读取 NIfTI 头检查三维尺寸和体素间距。SHA256 为下载后计算的本地校验值，并非官方提供的散列。

通过校验才将 `.part` 临时文件改为正式名称。续跑跳过已有且校验记录、路径、大小、修改时间一致的完整文件；未完成文件从该文件开头重试。已有正式文件若校验失败则保留待核查，不覆盖。MACOSX 资源分支不下载。

下载脚本：`scripts/download_amos_mm.py`。后台会话：`amos_mm_download`。

```bash
cd /xmlg/Lim/Project4/src
python scripts/download_amos_mm.py --workers 4
```

查看状态与日志：

```bash
cat /xmlg/Lim/Project4/outputs/amos_mm/download/status.json
tail -n 20 /xmlg/Lim/Project4/outputs/amos_mm/download/download.log
```

下载状态目录：`/xmlg/Lim/Project4/outputs/amos_mm/download/`。

- `archive_index.json`：官方 ZIP 文件目录、大小、偏移、CRC32。
- `data_mapping.json`：存储位置、原始元数据散列、重复文件和编号匹配结果。
- `candidate_image_paths.csv`：1,211 例候选编号对应的图像路径，图像是否已下载以状态和校验记录为准。
- `verified_files.jsonl`：逐文件完成记录、实际路径、SHA256、尺寸和体素间距。
- `status.json`：完成数量、字节数、速度、空闲空间和失败条目。

状态为 `complete` 且完成数为 `1687/1687` 时，图像下载与逐文件完整性检查才全部完成。
