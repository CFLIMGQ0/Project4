# show_watchResult.py 脚本说明

## 脚本用途
`scripts/show_watchResult.py` 用于统计 `valid_dicts_report.csv` 中 `watchResult` 的类型分布，并分别给出胃镜与肠镜的出现次数。

## 统计口径
- 通过 `reportTitle` 判断检查类型：
  - 含“胃”且不含“肠”归为胃镜；
  - 含“肠”归为肠镜。
- `watchResult` 会按中文/英文分隔符拆分（如 `，,；;。、“”` 等），然后统计细分类型出现频次。
- 脚本会输出进度条，便于批量数据扫描时查看处理进度。

## 使用方式
在仓库根目录运行：

```bash
python scripts/show_watchResult.py
```

可选参数：

```bash
python scripts/show_watchResult.py \
  --config configs/path.yaml \
  --report-csv /path/to/valid_dicts_report.csv \
  --output-csv /path/to/watch_result_summary.csv
```

## 输出结果
- 终端打印：
  - 胃镜 `watchResult` 细分结果（按次数降序）
  - 肠镜 `watchResult` 细分结果（按次数降序）
- CSV 文件（默认）：
  - `paths.output_dir/watch_result_summary.csv`
  - 字段：`organ, watchResult, count`
