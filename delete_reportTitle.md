# delete_reportTitle：胃镜 reportTitle 边界样本剔除分析说明

## 目标
本方案用于在**不训练模型**、仅基于现有 CSV 的条件下，对以下 3 个带 `*` 小类做“可解释剔除分析”：

- 胃镜下静脉曲张手术报告
- 急诊胃镜报告
- 超声胃镜下手术报告

## 输入文件
脚本默认读取以下 4 个文件（可通过参数覆盖）：

1. `valid_dicts_report.csv`
2. `centroid_cosine_similarity.csv`
3. `fid_matrix.csv`
4. `mmd_matrix.csv`

默认矩阵目录为：`/home/Lim/outputs/project4/check_similarity/gastric`（即 `check_similarity/gastric`）。

## 核心算法
采用“**多视图关系融合 + 边界样本裁剪**”：

1. **胃镜过滤**：仅保留 reportTitle 含“胃”且不含“肠”的记录。
2. **三视角同向化**：
   - cosine 原本“越大越相似”；
   - FID/MMD 原本“越小越相似”，先反向再归一化；
   - 三者统一到 `[0,1]`，都表示“越大越相似”。
3. **共识相似度融合**：
   - 默认等权：`(cos + fid + mmd) / 3`；
   - 支持 `--w-cos --w-fid --w-mmd` 调整权重。
4. **边界判别指标**：
   - `own_affinity`
   - `best_alt_affinity`
   - `margin`
   - `silhouette_like_score`
   - `purity_gain_cosine / purity_gain_fid / purity_gain_mmd`
5. **决策输出**：
   - 建议删除
   - 可删可留
   - 不建议删除

## 使用方式
在仓库根目录运行：

```bash
python scripts/analyze_star_report_titles.py
```

可选参数示例：

```bash
python scripts/analyze_star_report_titles.py \
  --similarity-dir /home/Lim/outputs/project4/check_similarity/gastric \
  --valid-dicts-report-csv /path/to/valid_dicts_report.csv \
  --centroid-cosine-csv /path/to/centroid_cosine_similarity.csv \
  --fid-matrix-csv /path/to/fid_matrix.csv \
  --mmd-matrix-csv /path/to/mmd_matrix.csv \
  --output-dir /path/to/output \
  --w-cos 1 --w-fid 1 --w-mmd 1
```

## 输出目录
脚本会在输出根目录下创建：

- `delete_reportTitle/star_report_title_analysis.csv`
- `delete_reportTitle/star_report_title_analysis.md`

并在终端打印：

- 各指标含义
- 3 个带 `*` 小类的关键数值与建议

## 结果解释建议
- 不以“样本少”作为唯一剔除依据。
- 若 `margin < 0` 且 `silhouette_like_score < 0`，且删除后三视角纯度明显改善，才倾向“建议删除”。
- 对“超声胃镜下手术报告”需额外讨论其**真实 hybrid subtype** 属性，不应简单按噪声处理。
