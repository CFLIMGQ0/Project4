"""完整训练的验证集研究任务：隔离测试集，保留可复查预测、梯度和配置。"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auto_research.model import ResearchModel
from training.data import _encode_text_fields
from training.losses import AsymmetricLossMultiLabel
from scripts.paper_block_deletion import block_sampling

DATASETS = {"ct_rate": "ct_rate_680", "amos_mm": "amos_mm", "mr_rate_1k": "mr_rate_1k"}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_ids(folder, dataset, fold):
    if dataset == "amos_mm":
        obj = json.loads((folder / "splits.json").read_text())
        excluded = {x["case_index"] for x in json.loads((folder / "image_exclusions.json").read_text())["excluded"]}
        validation = set(obj["validation_folds"][fold]) - excluded
        train = set(obj["development"]) - validation - excluded
        test = set(obj["test"])
    else:
        groups = json.loads((folder / "patient_folds.json").read_text())["folds"]
        # 固定保留第 0 组：后续深度搜索的验证折也不能进入它。
        test = set(groups[0])
        validation = set(groups[1 + fold])
        train = set().union(*groups[1:]) - validation
    assert not train & validation and not (train | validation) & test
    return sorted(train), sorted(validation), sorted(test)


class Bags(Dataset):
    def __init__(self, rows, ids, cache, config, training=False, ratio=0, blocks=1, reindexed=False):
        self.rows, self.ids, self.cache, self.config = rows, ids, cache, config
        self.training, self.ratio, self.blocks, self.reindexed = training, ratio, blocks, reindexed

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        index = self.ids[item]
        features, positions, count, token_ids, token_mask = self.cache[index]
        n = len(features)
        selected = np.arange(n)
        ratio, blocks = self.ratio, self.blocks
        if self.training:
            if self.config.get("block_training"):
                ratio = int(np.random.choice([0, 25, 50, 75]))
                blocks = int(np.random.choice([1, 3])) if ratio else 1
            else:
                keep = max(1, round(n * (1 - float(self.config.get("instance_dropout", .25)))))
                selected = np.sort(np.random.choice(n, keep, replace=False))
        if ratio:
            seed = int(np.random.randint(1, 1000000)) if self.training else 42
            selected = np.array(block_sampling(n, ratio, blocks, index, seed=seed)["selected_raw_indices"])
        pos = positions[selected]
        if self.reindexed:
            pos, count = np.arange(len(selected)), len(selected)
        row = self.rows[index]
        return (features[selected], pos, count, token_ids, token_mask,
                np.asarray(row["labels"], np.float32),
                np.asarray(row.get("known_mask", [True] * len(row["labels"])), bool), index)


def collate(rows):
    b, t = len(rows), max(len(row[0]) for row in rows)
    x = torch.zeros(b, t, 768, 1, 1)
    mask = torch.zeros(b, t, dtype=torch.bool)
    positions = torch.full((b, t), -1, dtype=torch.long)
    for j, row in enumerate(rows):
        n = len(row[0])
        x[j, :n, :, 0, 0] = torch.from_numpy(row[0])
        positions[j, :n] = torch.from_numpy(row[1])
        mask[j, :n] = True
    return dict(images=x, mask=mask, instance_indices=positions,
                original_image_counts=torch.tensor([r[2] for r in rows]),
                watch_token_ids=torch.stack([r[3] for r in rows]),
                watch_token_mask=torch.stack([r[4] for r in rows]),
                targets=torch.tensor(np.stack([r[5] for r in rows])),
                known=torch.tensor(np.stack([r[6] for r in rows])),
                ids=torch.tensor([r[7] for r in rows]))


def loss_fn(logits, targets, known, kind="asl"):
    if not known.any():
        return logits.sum() * 0
    if kind == "bce":
        return F.binary_cross_entropy_with_logits(logits[known].float(), targets[known].float())
    return AsymmetricLossMultiLabel()(logits[known], targets[known])


def label_f1(y, p, known, thresholds):
    predicted = p >= np.asarray(thresholds)
    y = y > .5
    tp = (predicted & y & known).sum(0)
    fp = (predicted & ~y & known).sum(0)
    fn = (~predicted & y & known).sum(0)
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def choose_thresholds(y, p, known):
    grid = np.arange(.1, .901, .05)
    scores = np.stack([label_f1(y, p, known, value) for value in grid])
    return grid[scores.argmax(0)]


def cuda_batch(batch):
    inputs = {k: v.cuda(non_blocking=True) for k, v in batch.items() if k not in {"targets", "known", "ids"}}
    return inputs, batch["targets"].cuda(), batch["known"].cuda()


def evaluate_steps(model, loader):
    model.eval()
    ys, ps, ks, ids = [], [], [], []
    total_loss, num = 0., 0
    agreement, coordinate_count = 0., 0
    for batch in loader:
        with torch.no_grad():
            inputs, target, known = cuda_batch(batch)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(**inputs)
            loss = loss_fn(out["logits"], target, known)
            total_loss += float(loss) * int(known.sum())
            num += int(known.sum())
            ys.append(target.cpu().numpy())
            ks.append(known.cpu().numpy())
            ps.append(out["logits"].float().sigmoid().cpu().numpy())
            ids.append(batch["ids"].numpy())
            if "apro_context_coordinates" in out:
                errors = (out["apro_context_coordinates"] - out["apro_raw_coordinates"]).abs()
                # 只是坐标相对输入锚点的偏离；不能当成真实位置恢复准确率。
                agreement += int(((errors <= .05) & inputs["mask"]).sum())
                coordinate_count += int(inputs["mask"].sum())
        yield {"phase": "evaluation"}
    return dict(y=np.concatenate(ys), p=np.concatenate(ps), known=np.concatenate(ks),
                ids=np.concatenate(ids), loss=total_loss / max(num, 1),
                anchor_agreement=agreement / coordinate_count if coordinate_count else None)


def gradient_diagnostic(model, terms):
    parameters = [p for n, p in model.named_parameters() if "apro_positioner.transition" in n and p.requires_grad]
    vectors, result = {}, {}
    for key, loss in terms.items():
        grad = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        vector = torch.cat([(g.float().flatten() if g is not None else torch.zeros_like(p).float().flatten())
                            for p, g in zip(parameters, grad)])
        vectors[key] = vector
        result[key + "_norm"] = float(vector.norm())
    for name in ("lqd", "image"):
        denominator = vectors["main"].norm() * vectors[name].norm()
        result["main_" + name + "_cosine"] = (float(torch.dot(vectors["main"], vectors[name]) / denominator)
                                                       if float(denominator) > 0 else None)
    return result


def run_job(args, shared_cache=None):
    spec = json.loads(args.job.read_text())
    config, dataset = spec["config"], spec["dataset"]
    fold, seed = spec.get("fold", 0), spec.get("seed", 42)
    epochs = int(config.get("epochs", 30))
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    folder = args.root / "outputs" / DATASETS[dataset] / "experiment"
    rows = {r["case_index"]: r for r in json.loads((folder / "samples.json").read_text())}
    train_ids, val_ids, test_ids = split_ids(folder, dataset, fold)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise RuntimeError("结果已经存在，拒绝覆盖")
    metadata = [folder / "samples.json", folder / ("splits.json" if dataset == "amos_mm" else "patient_folds.json")]
    source = Path(__file__).resolve().parents[1]
    source_files = list((source / "auto_research").glob("*.py")) + [source / "exp_8/models.py", source / "training/losses.py", source / "training/data.py"]
    protocol = dict(spec=spec, train_ids=train_ids, validation_ids=val_ids, excluded_test_ids=test_ids,
                    selection="验证集 ASL 最小的 checkpoint；干净验证集逐标签阈值用于全部删除条件",
                    deletion="对缓存的至多64张图像做block删除，非论文原体积删片协议；不读取测试特征",
                    historical_test_exposure="历史研究看过这些测试集；本轮隔离测试并不使其重新成为从未使用的外部集",
                    input_hashes={p.name: digest(p) for p in metadata},
                    code_hashes={str(p.relative_to(source)): digest(p) for p in source_files},
                    versions=dict(torch=torch.__version__, numpy=np.__version__), pid=os.getpid(),
                    execution_mode=getattr(args, "execution_mode", "standalone"))
    evaluation_resume = (output / "history.json").exists() and (output / "best.pt").exists()
    previous_history = json.loads((output / "history.json").read_text()) if evaluation_resume else []
    evaluation_resume = evaluation_resume and len(previous_history) == epochs
    if evaluation_resume:
        previous_protocol = json.loads((output / "protocol.json").read_text())
        if previous_protocol["spec"] != spec or previous_protocol["input_hashes"] != protocol["input_hashes"]:
            raise RuntimeError("已有训练与当前配置/数据不一致，不能复用 checkpoint")
        write_json(output / "evaluation_protocol.json", protocol)
    else:
        write_json(output / "protocol.json", protocol)
    cache_key = (str(folder.resolve()), tuple(sorted(set(train_ids + val_ids))))
    if shared_cache is not None and cache_key in shared_cache:
        cache, hashes = shared_cache[cache_key]
    else:
        cache, hashes = {}, {}
        for index in tqdm(sorted(set(train_ids + val_ids)), desc="读取训练/验证缓存", mininterval=5):
            path = folder / "features" / f"{index:04d}.npz"
            with np.load(path, allow_pickle=False) as z:
                feature = z["features"].astype(np.float32)
                positions = z["slice_indices"].astype(np.int64)
                count = int(z["original_count"])
            if not np.isfinite(feature).all() or len(feature) < 12:
                raise ValueError(f"缓存无法用于预注册删除条件：{index}")
            tokens, textmask = _encode_text_fields({"watch": rows[index]["findings_masked"]}, ("watch",), max_length=512, vocab_size=8192)
            cache[index] = (feature, positions, count, tokens, textmask)
            hashes[str(index)] = digest(path)
            if len(cache) % 64 == 0:
                yield {"phase": "loading", "loaded": len(cache)}
        if shared_cache is not None:
            shared_cache[cache_key] = (cache, hashes)
    write_json(output / "cache_hashes.json", hashes)
    model = ResearchModel(len(rows[train_ids[0]]["labels"]), config).cuda()
    # 新模块创建不会改变训练采样或 dropout 随机序列的起点。
    torch.manual_seed(seed + 1000)
    torch.cuda.manual_seed_all(seed + 1000)
    groups = [dict(params=[p for n, p in model.named_parameters() if "apro_positioner" not in n], lr=float(config.get("lr", 2e-4))),
              dict(params=[p for n, p in model.named_parameters() if "apro_positioner" in n], lr=float(config.get("lr", 2e-4)) * float(config.get("position_lr_multiplier", 1)))]
    optimizer = torch.optim.AdamW(groups, weight_decay=float(config.get("weight_decay", .02)))
    train_loader = DataLoader(Bags(rows, train_ids, cache, config, training=True), batch_size=16,
                              shuffle=True, num_workers=0, collate_fn=collate)
    eligible_ids = []
    for index in val_ids:
        try:
            for ratio, blocks in ((50, 1), (80, 1), (80, 8)):
                block_sampling(len(cache[index][0]), ratio, blocks, index)
        except ValueError:
            continue
        eligible_ids.append(index)
    if not eligible_ids:
        raise RuntimeError("验证集没有同时满足全部删除条件的图像包")
    write_json(output / "deletion_eligibility.json", dict(eligible_ids=eligible_ids,
               excluded_ids=sorted(set(val_ids)-set(eligible_ids)),
               reason="全部删除条件共用可构造80%/8个非相邻非空块的样本；另报同子集0%结果"))
    def loader(ratio=0, blocks=1, reindexed=False, eligible=False):
        selected_ids = eligible_ids if ratio or eligible else val_ids
        return DataLoader(Bags(rows, selected_ids, cache, config, ratio=ratio, blocks=blocks, reindexed=reindexed),
                          batch_size=16, shuffle=False, num_workers=0, collate_fn=collate)
    scaler = torch.amp.GradScaler("cuda")
    total_steps = epochs * len(train_loader)
    warmup = max(1, int(.2 * total_steps))
    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        return .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    best, best_epoch, started = float("inf"), -1, time.time()
    history = previous_history if evaluation_resume else []
    if evaluation_resume:
        print("完整训练已经完成，复用已保存权重，仅恢复删除评估", flush=True)
        best_epoch = int(torch.load(output / "best.pt", map_location="cpu", weights_only=True)["epoch"])
    yield {"phase": "initialized"}
    for epoch in range(0 if evaluation_resume else epochs):
        model.train()
        model.epoch = epoch
        running = dict(main=0., image=0., lqd=0., total=0., lqd_weight=0.)
        diagnostics = None
        for step, batch in enumerate(train_loader):
            inputs, target, known = cuda_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(**inputs, labels=(target, known))
            main_loss = loss_fn(out["logits"], target, known, config.get("classification_loss", "asl"))
            image_loss = loss_fn(out["image_only_logits"], target, known, config.get("image_loss", "asl"))
            lqd_loss = out["research_lqd"]
            image_weight = float(config.get("image_weight", 0))
            lqd_weight = float(config.get("lqd_weight", .01)) if epoch >= config.get("lqd_delay", 0) else 0
            if config.get("lqd_decay"):
                lqd_weight *= .5 * (1 + math.cos(math.pi * epoch / max(1, epochs-1)))
            step_diagnostics = None
            if config.get("lqd_gradient_cap") and lqd_weight:
                # 只借助 ACPE 转变参数的梯度控制辅助目标占比，不强行等大模态梯度。
                step_diagnostics = gradient_diagnostic(model, dict(main=main_loss, image=image_loss, lqd=lqd_loss))
                bound = float(config["lqd_gradient_cap"]) * step_diagnostics["main_norm"] / max(step_diagnostics["lqd_norm"], 1e-12)
                lqd_weight = min(lqd_weight, bound)
            if config.get("visual_warmup", 0) > epoch:
                total = image_loss
            else:
                total = main_loss + image_weight * image_loss + lqd_weight * lqd_loss
            if config.get("coordinate_penalty", 0):
                delta = out["apro_context_coordinates"] - out["apro_raw_coordinates"]
                total = total + config["coordinate_penalty"] * delta[inputs["mask"]].square().mean()
            if step == 0 and epoch in {0, 4, 9, 19, epochs-1}:
                diagnostics = step_diagnostics or gradient_diagnostic(model, dict(main=main_loss, image=image_loss, lqd=lqd_loss))
            if not torch.isfinite(total):
                raise FloatingPointError(f"非有限损失：epoch={epoch + 1}")
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            for key, value in dict(main=main_loss, image=image_loss, lqd=lqd_loss, total=total).items():
                running[key] += float(value.detach())
            running["lqd_weight"] += lqd_weight
            yield {"phase": "training", "epoch": epoch+1, "step": step+1, "steps": len(train_loader)}
        val = yield from evaluate_steps(model, loader())
        row = dict(epoch=epoch+1, train={k: v / len(train_loader) for k, v in running.items()},
                   validation_loss=val["loss"], validation_f1_05=float(label_f1(val["y"], val["p"], val["known"], .5).mean()),
                   gradient=diagnostics, seconds=time.time()-started,
                   peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
        if val["loss"] < best:
            best, best_epoch = val["loss"], epoch + 1
            temporary = output / "best.tmp"
            torch.save(dict(model=model.state_dict(), epoch=best_epoch, config=config), temporary)
            temporary.replace(output / "best.pt")
        history.append(row)
        write_json(output / "history.json", history)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    saved = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model"])
    model.epoch = best_epoch - 1
    clean = yield from evaluate_steps(model, loader())
    thresholds = choose_thresholds(clean["y"], clean["p"], clean["known"])
    conditions, predictions = {}, {}
    for name, ratio, blocks, reindex in [("clean",0,1,False), ("clean_eligible",0,1,False), ("d50_b1",50,1,False), ("d80_b1",80,1,False),
                                         ("d80_b8",80,8,False), ("d80_b1_reindexed",80,1,True)]:
        val = clean if name == "clean" else (yield from evaluate_steps(model, loader(ratio, blocks, reindex, eligible=name=="clean_eligible")))
        conditions[name] = dict(macro_f1=float(label_f1(val["y"],val["p"],val["known"],thresholds).mean()),
                                macro_f1_05=float(label_f1(val["y"],val["p"],val["known"],.5).mean()),
                                loss=val["loss"], anchor_agreement=val["anchor_agreement"], cases=len(val["ids"]))
        for key in ("y", "p", "known", "ids"):
            predictions[name + "_" + key] = val[key]
    np.savez_compressed(output / "validation_predictions.npz", **predictions)
    result = dict(spec=spec, best_epoch=best_epoch, thresholds=thresholds.tolist(), conditions=conditions,
                  peak_reserved_mib=max(torch.cuda.max_memory_reserved()/2**20, max(r["peak_reserved_mib"] for r in history)),
                  seconds=history[-1]["seconds"]+time.time()-started if evaluation_resume else time.time()-started,
                  status="completed", evaluation_scope="validation_only_cached_bag_screening",
                  execution_mode=getattr(args, "execution_mode", "standalone"),
                  memory_accounting_scope="shared_process" if shared_cache is not None else "single_job")
    write_json(output / "result.json", result)
    print("任务完成：" + str(output / "result.json"), flush=True)


def evaluate(model, loader):
    iterator = evaluate_steps(model, loader)
    while True:
        try:
            next(iterator)
        except StopIteration as finished:
            return finished.value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    for _ in run_job(args):
        pass


if __name__ == "__main__":
    main()
