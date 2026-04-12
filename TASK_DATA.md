# 任务数据筛选结果

## 胃镜三标签多标签任务

本次共识别出 `2898` 条胃镜记录，其中 `2863` 条被纳入多标签任务，累计图像 `171642` 张。

标签分布：

- 食管 SMT（字段：`label_esophageal_smt`）：`1375`
- 食管黏膜病变 / 食管肿物（字段：`label_esophageal_mucosal_or_tumor`）：`1489`
- 胃炎类（字段：`label_gastritis`）：`1199`

主要剔除原因：

- `watchResult` 为空：`2`
- 仅出现不确定表述，未命中目标标签：`10`
- 未命中目标标签：`23`

## 肠镜二分类任务

本次共识别出 `548` 条肠镜记录，其中 `507` 条被纳入二分类任务，累计图像 `39260` 张。

类别分布：

- 正常（`0`）：`213`
- 息肉（`1`）：`294`

主要剔除原因：

- 未命中目标标签：`41`

## 输出文件

生成文件位于：`/home/Lim/Project4/datasets/task_data`

- `gastro_multilabel_task_datalist.csv`
- `colonoscopy_binary_task_datalist.csv`

训练图像缓存也约定放在同一目录下，并按任务拆分：

- `cache_gastro_multilabel_image/`：仅胃镜任务训练时使用与维护。
- `colonoscopy_binary_image_cache/`：仅肠镜任务训练时使用与维护。

说明：

- 当前若只运行胃镜模型，则只会预构建并复用 `cache_gastro_multilabel_image/`；
- 当前若只运行肠镜模型，则只会预构建并复用 `colonoscopy_binary_image_cache/`；
- 两类任务缓存互不混用，但都会长期保留在 `task_data` 目录下，便于后续重复训练直接复用。
- 当前磁盘缓存保存的是预缩放后的 `uint8 RGB numpy`，不是原始分辨率图像；以当前 `image_size: 224` 为例，缓存实际尺寸为 `336x336`。
- 写缓存时会先按短边缩放，再做中心裁剪，并把缓存签名与 `cache_image_size` 绑定，避免不同输入尺寸复用同一份旧缓存。
- 按当前策略粗略估算，单张缓存约 `339 KB`；若先做 `500` 个 exam 的小样本探索，首轮缓存约 `10.8 GB`，全量缓存约 `72 GB`。
