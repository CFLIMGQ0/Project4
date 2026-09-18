#!/usr/bin/env python3
"""保留既有特征协议，按检查分片并行提取；单例损坏独立记录。"""
import argparse
import fcntl
import json
from pathlib import Path
import time
import traceback

import numpy as np
from tqdm import tqdm

import prepare_amos_mm_experiments as base

POOL = base.OUT / "parallel_features"


def valid_cache(path, row, digest):
    with np.load(path, allow_pickle=False) as data:
        assert str(data["scan_id"]) == row["scan_id"] and str(data["preparation_sha256"]) == digest
        assert data["features"].shape == (len(data["slice_indices"]), 768) and np.isfinite(data["features"]).all()
        p = data["slice_indices"]
        assert np.all(np.diff(p) > 0) and 0 <= p[0] <= p[-1] < int(data["original_count"])


def worker(shard, shards, cases=None):
    import nibabel as nib
    import torch
    import torch.nn.functional as F
    from torchvision import models
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.12)
    torch.hub.set_dir(str(base.ROOT / "pre_weights"))
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    encoder = torch.nn.Sequential(model.features, model.avgpool, model.classifier[0], torch.nn.Flatten(1)).eval().cuda()
    del model
    mean = torch.tensor([.485, .456, .406], device="cuda")[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device="cuda")[None, :, None, None]
    protocol = json.loads((base.OUT / "preparation_protocol.json").read_text())
    assert base.sha256(Path(base.__file__)) == protocol["preparation_source_sha256"]
    assert protocol["windows_level_width"] == [list(v) for v in base.WINDOWS]
    digest = protocol["preparation_sha256"]
    rows = json.loads((base.OUT / "samples.json").read_text())
    jobs = [r for r in rows if r["case_index"] in cases] if cases else [r for r in rows if r["case_index"] % shards == shard]
    folder = base.OUT / "features"
    folder.mkdir(parents=True, exist_ok=True)
    begin = time.monotonic()
    errors = []
    for step, row in enumerate(tqdm(jobs, desc=f"分片{shard}提取完整轴位特征"), 1):
        path = folder / f"{row['case_index']:04d}.npz"
        try:
            with path.with_suffix(".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if path.exists():
                    valid_cache(path, row, digest)
                    continue
                image = nib.load(str(base.DATA / row["file_path"]))
                assert len(image.shape) == 3 and np.isfinite(image.affine).all()
                data = np.asarray(image.dataobj, dtype=np.float32)
                orient = nib.orientations.ornt_transform(nib.orientations.io_orientation(image.affine),
                                                        nib.orientations.axcodes2ornt(("R", "A", "S")))
                data = nib.orientations.apply_orientation(data, orient)
                total = data.shape[2]
                indices = np.unique(np.linspace(0, total - 1, min(64, total)).round().astype(np.int64))
                slices = np.ascontiguousarray(data[::-1, ::-1, indices].transpose(2, 1, 0))
                del data
                assert np.isfinite(slices).all()
                values_out = []
                with torch.inference_mode():
                    for start in range(0, len(slices), 16):
                        values = torch.from_numpy(slices[start:start + 16]).cuda()[:, None]
                        channels = torch.cat([((values - (l - w / 2)) / w).clamp(0, 1) for l, w in base.WINDOWS], dim=1)
                        channels = F.interpolate(channels, (224, 224), mode="bilinear", align_corners=False, antialias=True)
                        with torch.autocast("cuda"):
                            feat = encoder((channels - mean) / std)
                        values_out.append(feat.float().cpu().numpy())
                result = np.concatenate(values_out)
                assert result.shape == (len(indices), 768) and np.isfinite(result).all()
                temp = path.with_suffix(".tmp.npz")
                np.savez_compressed(temp, features=result, slice_indices=indices, original_count=total,
                                    scan_id=row["scan_id"], preparation_sha256=digest,
                                    hu_quantiles=np.quantile(slices, [0, .01, .5, .99, 1]))
                temp.replace(path)
                valid_cache(path, row, digest)
        except Exception:
            error = {"case_index": row["case_index"], "scan_id": row["scan_id"], "traceback": traceback.format_exc(), "time": time.time()}
            errors.append(error)
            base.save_json(POOL / "errors" / f"{row['scan_id']}.json", error)
            print(error, flush=True)
        finally:
            base.save_json(POOL / f"shard_{shard}_status.json", {"state": "running", "completed": step,
                           "total": len(jobs), "errors": len(errors), "updated_at": time.time()})
    base.save_json(POOL / f"shard_{shard}_status.json", {"state": "finished", "completed": len(jobs),
                   "errors": errors, "elapsed_seconds": time.monotonic() - begin, "updated_at": time.time()})


def monitor():
    rows = json.loads((base.OUT / "samples.json").read_text())
    digest = json.loads((base.OUT / "preparation_protocol.json").read_text())["preparation_sha256"]
    checked = set()
    while True:
        for row in rows:
            i = row["case_index"]
            path = base.OUT / "features" / f"{i:04d}.npz"
            if i not in checked and path.exists():
                valid_cache(path, row, digest)
                checked.add(i)
        missing = [r["scan_id"] for r in rows if r["case_index"] not in checked]
        base.save_json(base.OUT / "feature_status.json", {"state": "complete" if not missing else "running",
                       "completed": len(checked), "total": len(rows), "missing": missing,
                       "preparation_sha256": digest, "updated_at": time.time()})
        if not missing:
            print("1687例完整缓存校验通过，训练队列可以继续。", flush=True)
            break
        time.sleep(15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--cases", type=int, nargs="+")
    parser.add_argument("--monitor", action="store_true")
    args = parser.parse_args()
    monitor() if args.monitor else worker(args.shard, args.shards, args.cases)
