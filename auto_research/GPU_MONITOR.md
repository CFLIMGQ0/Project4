# 每卡单进程运行与实验监控

本轮研究使用 `controller_single.py` 调度：每张允许使用的 GPU 只启动一个 `gpu_worker.py` CUDA 进程。多个实验在同一进程中按完整 batch 交错执行，各自保留模型、优化器、随机状态、日志、checkpoint 和结果。GPU 0 继续禁用；其他用户的进程保持原样，因此全卡的进程总数可能大于一。

在 `/xmlg/Lim/Project4` 下运行：

```bash
python3 src/auto_research/gpu_view.py -i
```

每两秒刷新 204 和 202 的 GPU 状态；每张卡显示一个 `ALM-MIL [进程数/内部任务数, 总显存]` 显示组。全卡利用率、全卡显存与其他计算进程数仍真实显示。两台主机 GPU 0 的禁用标记读取当前资源策略。

展开内部任务、PID 和已完成 epoch：

```bash
python3 src/auto_research/gpu_view.py -i --details
```

只看 204，或仅打印一次：

```bash
python3 src/auto_research/gpu_view.py -i --hosts 204
python3 src/auto_research/gpu_view.py
```

原生 `gpustat` / `nvidia-smi` 按真实 PID 显示；新模式下每卡只有一个本轮研究的 GPU 进程。监控脚本从该 worker 的状态文件读取内部实验。组显存是 NVIDIA 报告的本轮计算进程显存之和，可能因驱动开销及其他程序而小于全卡已用显存。

监控脚本仅用 Python 标准库；202 的只读采样使用本项目现有 SSH 密钥。无需启用 MPS。切换时先让原有多进程任务自然完成，然后自动使用新的单进程调度。

训练任务仍按原配置运行完整 30 个 epoch。进程根据实际可用显存接收内部任务，同进程共享只读 CPU 数据缓存；显存峰值在结果中标记为整个共享进程的统计值，不能解释成某一模型的独占显存。
