# check_task_definitions 任务清单（临时）

## 目标
在不训练模型、不下载外部数据的前提下，基于现有 `valid_dicts_report.csv` 生成胃镜/肠镜任务标签文件与统计报告。

## 待办步骤
1. 读取 `configs/path.yaml`，定位 `valid_dicts_report.csv`。
2. 解析 CSV 字段并做字段名鲁棒映射（`reportTitle/watchResult/exam_dir/img_num`）。
3. 对文本做标准化预处理（NFKC、空格清理、全半角/括号统一）。
4. 按规则识别胃镜与肠镜样本范围。
5. 生成胃镜 3 标签多标签结果与剔除原因。
6. 生成肠镜二分类结果与剔除原因。
7. 生成肠镜三分类候选结果与剔除原因。
8. 统计样本数、图像数、标签分布与剔除分布。
9. 输出 3 个任务 CSV 与 `task_definition_summary.md`。
10. 在终端打印关键统计信息与输出路径。
