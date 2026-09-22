# MR-RATE六卡任务队列恢复（2026-09-19）

## 停滞原因

- 旧调度器固定按GPU编号分配模型/折次分片。本机GPU0/1显存长期不足时，其分片不能转移到其他卡。
- 本机GPU0/1和202主机GPU0/1共用以整数GPU编号为键的重试计数，不同主机的失败会互相影响。计数超过上限后停止分配，但整体状态仍显示运行。
- GPU3已经跑完自己的分片，却被重复拉起，不产生新结果。
- AMEF四标签训练的标签查询判别损失仍固定使用三标签置换，导致四标签视觉token和三标签替换文本token维度不匹配。

## 修复及验证

- 标签置换改为按`num_labels`生成全部非恒等循环置换，不新增模块或更换模型结构。四标签得到三个循环置换；三标签仍是原来的两个。
- 已用真实MR缓存检查四标签前向、损失和反向梯度；并用修复前源码核对三标签logits和所有辅助损失逐位一致。证据在`outputs/mr_rate_1k/all_models_fivefold/four_label_recovery_smoke.json`。
- 调度按未完成的“模型＋折次”分配，不再绑定固定GPU。监测本机四张卡和202两张卡，达到7500MiB剩余显存即参与分配；允许同卡多个任务，启动初期预留显存，避免读取尚未实际分配的空闲值而过量启动。
- 重试计数按任务独立记录，最多尝试三次；一个任务失败不会禁用整张卡。只剩失败任务时输出不完整状态，不再假装一直运行。
- 完成的远程任务逐个同步回本机后计数；完成集合不对两台主机的重复副本重复计数。每次任务有独立日志及调度历史。
- 原始磁盘和MR特征不删除、不重提；启动时检查本机和202结果盘至少有3GiB剩余空间。

## 保留旧结果的边界

已完成的66个任务属于未受此次文本查询损失修复影响的20种基线，AMEF多模态没有已完成结果。逐个核对完成标记、原协议哈希和200例四标签测试预测后保留这66个结果。

`recovery_protocol.json`记录兼容白名单、已完成任务清单、原指标与预测文件哈希，以及改变的源码。白名单明确排除`amef_multimodal`。患者、五折、特征、文本、超参数、种子和阈值选择均未变化。

旧完成标记和指标中的协议哈希保持原样，不伪装成新代码训练结果。共用聚合器仅在调用方显式传入逐模型兼容白名单时接受旧协议，其他实验默认仍要求严格相同协议。恢复前协议和源码保存在`protocol.pre_label_query_recovery.json`及`source_before_label_query_recovery/`。

## 运行及查看

```bash
python -u src/scripts/run_mrrate_1k_gpu_pool.py
```

状态：`outputs/mr_rate_1k/all_models_fivefold/gpu_pool_status.json`。

日志：`outputs/mr_rate_1k/all_models_fivefold/gpu_queue_recovery.log`、`gpu_pool_logs/`、`gpu_job_history_*.json`。

最终表格：`outputs/mr_rate_1k/table2_results.md`；完整数值在`all_models_fivefold/summary.csv`及`summary.json`。旧错误日志保留作审计，应结合新任务日志与完成状态判断是否仍在报错。
