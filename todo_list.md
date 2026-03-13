# 本次数据核验代办清单

## 目标
- 核验 `all_patients_raw.xlsx` 中患者姓名与数据集患者目录是否一致。

## 待办事项
- [ ] 确认数据集根目录路径（目录下每个患者一个文件夹，且含空的 `report.xlsx`）。
- [ ] 确认 `all_patients_raw.xlsx` 的文件路径与姓名列名。
- [ ] 运行 `scripts/data_cleaning/check_data.py` 做姓名一致性检查。
- [ ] 在终端查看 `scripts/data_cleaning/check_data.py` 的核验输出结果。
- [ ] 根据报告处理“仅 Excel 存在”与“仅目录存在”的姓名差异。
- [ ] 差异处理后再次运行检查，直到姓名集合完全一致。
