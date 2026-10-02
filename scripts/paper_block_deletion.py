#!/usr/bin/env python3
"""论文删片实验共用的 B-block 随机掩码；B=3 与旧实现逐样本一致。"""

from __future__ import annotations

import math

import numpy as np

SEED = 42
TARGET = 64
RATIOS = tuple(range(0, 81, 10))
BLOCKS = tuple(range(1, 9))


def block_sampling(total: int, ratio: int, blocks: int, case_index: int,
                   *, seed: int = SEED, target: int = TARGET) -> dict:
    if not 0 <= ratio <= 80 or blocks not in BLOCKS or total <= 0:
        raise ValueError("非法删除配置")
    if ratio == 0 and blocks != 1:
        raise ValueError("0% 删除只运行一次；其他 B 仅在绘图时复制同一基线")
    delete_count = math.floor(total * ratio / 100)
    remaining_count = total - delete_count
    if remaining_count < 3:
        raise ValueError(f"删除后不足3张，无法计算内部位置恢复指标：N={total}, ratio={ratio}")
    if ratio > 0 and (delete_count < blocks or remaining_count < blocks + 1):
        raise ValueError(f"无法构造 {blocks} 个非空非相邻块：N={total}, D={delete_count}")
    deleted_mask = np.zeros(total, dtype=bool)
    segments = []
    if delete_count:
        rng = np.random.default_rng(np.random.SeedSequence([seed, case_index, ratio, blocks]))
        base, remainder = divmod(delete_count, blocks)
        lengths = [base + int(i < remainder) for i in range(blocks)]
        rng.shuffle(lengths)
        cuts = np.sort(rng.choice(np.arange(1, remaining_count), size=blocks, replace=False))
        keeps = np.diff(np.concatenate(([0], cuts, [remaining_count]))).astype(int)
        cursor = 0
        for segment, length in enumerate(lengths):
            cursor += int(keeps[segment])
            start, end = cursor, cursor + int(length) - 1
            deleted_mask[start:end+1] = True
            segments.append({"segment": segment, "start": start, "end_inclusive": end,
                             "length": int(length)})
            cursor = end + 1
        assert cursor + int(keeps[-1]) == total
    remaining = np.flatnonzero(~deleted_mask)
    sampled_count = min(target, len(remaining))
    slots = np.rint(np.linspace(0, len(remaining)-1, sampled_count)).astype(np.int64)
    selected = remaining[slots]
    assert len(np.unique(selected)) == sampled_count and selected[0] == 0 and selected[-1] == total-1
    assert int(deleted_mask.sum()) == delete_count
    assert all(right["start"] > left["end_inclusive"]+1 for left, right in zip(segments, segments[1:]))
    return {"seed": seed, "case_index": case_index, "ratio": ratio, "blocks": blocks,
            "source_count": total, "deleted_count": delete_count,
            "deleted_raw_indices": np.flatnonzero(deleted_mask).tolist(),
            "segments": segments, "selected_raw_indices": selected.tolist(),
            "selected_remaining_slots": slots.tolist(), "target_instances": target,
            "valid_instances": sampled_count}


def feasible_configs(total: int, case_index: int, *, target: int = TARGET):
    for ratio in RATIOS:
        for blocks in ((1,) if ratio == 0 else BLOCKS):
            try:
                yield block_sampling(total, ratio, blocks, case_index, target=target)
            except ValueError:
                continue
