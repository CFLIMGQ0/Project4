# exp11_module_ablation 实验说明

`exp11_module_ablation` 用于补齐四模块全因子消融中尚未运行的10个组合。

## 模块定义

- M1：位置编码 + Transformer context encoder。
- M2：标签超图学习；关闭时使用普通 `learnable label graph`。
- M3：label-wise cross-attention。
- M4：文本融合门控。

当 M3、M4 均关闭时只输入图像；只启用 M3 时使用 cross-attention 后直接相加；
只启用 M4 时使用 pooled watch embedding + gate；同时启用 M3、M4 时使用
cross-attention + gate。

## 本轮10个实验

| 实验名 | 组合 | M1 | M2 | M3 | M4 |
|---|---|---:|---:|---:|---:|
| `exp11_module_ablation_none` | `∅` | × | × | × | × |
| `exp11_module_ablation_1` | `1` | ✓ | × | × | × |
| `exp11_module_ablation_2` | `2` | × | ✓ | × | × |
| `exp11_module_ablation_3` | `3` | × | × | ✓ | × |
| `exp11_module_ablation_4` | `4` | × | × | × | ✓ |
| `exp11_module_ablation_13` | `13` | ✓ | × | ✓ | × |
| `exp11_module_ablation_14` | `14` | ✓ | × | × | ✓ |
| `exp11_module_ablation_23` | `23` | × | ✓ | ✓ | × |
| `exp11_module_ablation_24` | `24` | × | ✓ | × | ✓ |
| `exp11_module_ablation_34` | `34` | × | × | ✓ | ✓ |

所有实验固定输入最多64张原图、使用种子2026，并沿用现有 `exp_9` 的
`image_aux_weight: 0.5`。纯图像组合不产生独立 image-only 辅助分支，因此该权重
只对使用 watch 文本融合的组合生效，与已有组合的训练口径一致。结果写入：

续训资源分配：组合 `14` 使用GPU `[0, 1, 2]`；组合 `23`、`24`、`34`
使用GPU `[0, 1, 2, 3]`。其他后续实验默认使用4张GPU。

```text
/home/Lim/Project4/outputs/train_runs/task2/exp11_module_ablation/
```

## 一键运行

当前 `configs/task2/train.yaml` 已开启 `auto_exp_11_module_ablation: true`，在 `src`
目录执行：

```bash
python train.py
```

即可顺序运行10个实验。指定部分实验时可使用：

```bash
python train.py --models exp11_module_ablation_none,exp11_module_ablation_1
```

tmux 后台运行命令：

```bash
./scripts/run_exp11_module_ablation_tmux.sh
```

查看运行日志：

```bash
tmux attach -t exp11_module_ablation
```
