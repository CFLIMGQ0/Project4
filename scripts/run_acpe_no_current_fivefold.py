#!/usr/bin/env python3
"""补齐表3的23456+1五折：仅移除当前特征拼接，沿用原消融训练接口。"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from prepare_abdomenatlas3_table1 import save_json, sha256

VARIANT = "23456+1"
GROUPS = [2, 3, 4, 5, 6]
MODEL = "amef_multimodal"
OUT = ROOT / "outputs/public_transition_no_current_fivefold"
ROOTS = {"ct_rate": ROOT / "outputs/ct_rate_680/acpe_transition_groups_fivefold",
         "mr_rate": ROOT / "outputs/mr_rate_1k/transition_groups",
         "amos_mm": ROOT / "outputs/amos_mm/transition_groups_7_labels"}
NAMES = {"ct_rate": "CT-RATE", "mr_rate": "MR-RATE-1K", "amos_mm": "AMOS-MM"}


def folder(dataset, fold):
    root = ROOTS[dataset] / VARIANT
    if dataset == "amos_mm":
        root /= "7_labels"
    return root / f"fold_{fold}" / MODEL


def extend_protocol(value):
    value.pop("protocol_sha256", None)
    value["source_sha256"][str(Path(__file__).relative_to(ROOT))] = sha256(Path(__file__))
    value["target_row"] = "表3：移除u_it，保留前后差分、幅值、交互项及采集间距"
    value["controlled_change"] = "五组64维加间距，共321维；真实删除输入项，不补零；无r051位置注入预热；其余沿用历史描述子消融"
    value["protocol_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return value


def checked_builder(original):
    def build(*args, **kwargs):
        model, weights = original(*args, **kwargs)
        modules = [m for m in model.modules() if hasattr(m, "transition_groups")]
        assert len(modules) == 1
        acpe = modules[0]
        assert tuple(acpe.transition_groups) == tuple(GROUPS)
        assert acpe.transition_include_gap and acpe.transition_mlp[1].in_features == 321
        print("已核对ACPE描述子：23456+1，321维", flush=True)
        return model, weights
    return build


def configure(dataset):
    if dataset == "ct_rate":
        import run_ctrate_acpe_transition_groups as exp
        exp.VARIANTS = {VARIANT: (GROUPS, True)}
        original = exp.protocol
        exp.protocol = lambda: extend_protocol(original())
        exp.configure(VARIANT)
        exp.core.build_model = checked_builder(exp.core.build_model)
        return exp, exp.core, exp.protocol()
    import run_amos_mrrate_transition_groups as exp
    exp.VARIANTS = {VARIANT: (GROUPS, True)}
    exp.DISPLAY_NAMES = {VARIANT: "ACPE transition 23456+1"}
    original = exp.protocol
    exp.protocol = lambda data, variant: extend_protocol(original(data, variant))
    original_amos, original_mr = exp.configure_amos, exp.configure_mr
    def configure_amos(variant):
        core = original_amos(variant)
        core.build_model = checked_builder(core.build_model)
        return core
    def configure_mr(variant):
        core = original_mr(variant)
        core.build_model = checked_builder(core.build_model)
        core.configure_base()
        return core
    exp.configure_amos, exp.configure_mr = configure_amos, configure_mr
    exp.install_base_hooks()
    core = exp.configure_amos(VARIANT) if dataset == "amos_mm" else exp.configure_mr(VARIANT)
    return exp, core, exp.protocol(dataset, VARIANT)


def initialize(dataset):
    exp, core, p = configure(dataset)
    params = exp.parameters()[MODEL] if dataset == "ct_rate" else exp.params_for(VARIANT)
    reference_root = ROOTS[dataset] / "123+1"
    configs = []
    for fold in range(1, 6):
        previous = reference_root
        if dataset == "amos_mm":
            previous /= "7_labels"
        previous /= f"fold_{fold}"
        previous /= MODEL
        old = json.loads((previous / "config.json").read_text())
        field = "parameters" if dataset == "amos_mm" else "model"
        reference = dict(old[field])
        reference["apro_transition_groups"] = GROUPS
        assert params == reference, f"{dataset}折{fold}消融以外模型参数改变"
        assert core.SETTINGS == old["settings"], f"{dataset}训练设置改变"
        assert old["seed"] == 42 + 100*fold
        configs.append({"fold":fold, "config_sha256":sha256(previous / "config.json")})
    p["reference_config_audit"] = configs
    # 审计信息独立保存，不改变训练器自身校验的协议。
    audit = p.pop("reference_config_audit")
    path = ROOTS[dataset] / VARIANT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == p, f"{path}协议改变"
    else:
        save_json(path,p)
    save_json(OUT / f"audit_{dataset}.json", {"dataset":dataset,"input_dim":321,
        "reference_configs":audit,"settings":core.SETTINGS,"protocol_sha256":p["protocol_sha256"]})
    print(f"{NAMES[dataset]}：五折参数核对通过，输入321维",flush=True)
    return exp,core,p


def verify_fold(dataset, fold, p):
    target = folder(dataset,fold)
    reference = Path(str(target).replace('/'+VARIANT+'/', '/123+1/'))
    config = json.loads((target / "config.json").read_text())
    old = json.loads((reference / "config.json").read_text())
    assert config["settings"] == old["settings"] and config["seed"] == old["seed"]
    if dataset == "amos_mm":
        for key in ("train","validation","test","image_exclusions_applied"):
            assert config[key] == old[key], (dataset,fold,key)
        metric = json.loads((target / "result.json").read_text())
        required = ["best_model.pt","test_predictions.npz","history.json"]
    else:
        assert json.loads((target / "split_ids.json").read_text()) == json.loads((reference / "split_ids.json").read_text())
        metric = json.loads((target / "test_metrics.json").read_text())
        required = ["completed.json","best_model.pt","test_predictions.csv","validation_predictions.csv","history.json"]
    assert metric["protocol_sha256"] == p["protocol_sha256"]
    assert all((target / name).exists() for name in required)
    save_json(OUT / "completed" / f"{dataset}_fold_{fold}.json", {
        "dataset":dataset,"fold":fold,"protocol_sha256":p["protocol_sha256"],
        "macro_f1":metric["macro_f1"],"source":str(target),"splits_match":True})


def summarize():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT / "summary.lock").open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        rows=[]
        lines=["# 表3：移除当前特征u_it的五折实验", "", "仅改变描述子为23456+1（321维）；保持原数据、文本、划分、训练设置和验证选模。未增加位置预热。", "",
               "| 数据集 | 完成折数 | 测试F1（%，均值±样本标准差） |", "|---|---:|---:|"]
        for dataset in ROOTS:
            records=[json.loads(p.read_text()) for p in sorted((OUT/'completed').glob(f'{dataset}_fold_*.json'))]
            row={"dataset":dataset,"completed_folds":len(records),"folds":records}
            cell='-'
            if len(records)==5:
                assert len({r['protocol_sha256'] for r in records})==1
                row.update(macro_f1_mean=statistics.mean(r['macro_f1'] for r in records),
                           macro_f1_std=statistics.stdev(r['macro_f1'] for r in records))
                cell=f"{100*row['macro_f1_mean']:.2f} ± {100*row['macro_f1_std']:.2f}"
                save_json(ROOTS[dataset]/VARIANT/'summary.json',row)
            rows.append(row)
            lines.append(f"| {NAMES[dataset]} | {len(records)}/5 | {cell} |")
        lines += ["", "AMOS-MM沿用固定200例人工测试集、五个开发折训练；CT-RATE与MR-RATE沿用原五折测试划分。", "结果保存在各数据集的23456+1目录，论文不自动改动。", ""]
        save_json(OUT / "summary.json",rows)
        (OUT/'results.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=ROOTS)
    parser.add_argument('--fold',type=int,choices=range(1,6))
    parser.add_argument('--initialize',action='store_true')
    parser.add_argument('--manifest',action='store_true')
    args=parser.parse_args()
    if args.manifest:
        jobs=[]
        for fold in range(1,6):
            for dataset in ROOTS:
                p=json.loads((ROOTS[dataset]/VARIANT/'protocol.json').read_text())
                jobs.append({"kind":"external","key":f"no_current_{dataset}_fold_{fold}",
                    "dataset":dataset,"fold":fold,"protocol_sha256":p['protocol_sha256'],
                    "completion_path":str(OUT/'completed'/f'{dataset}_fold_{fold}.json'),
                    "arguments":[str(Path(__file__).resolve()),'--dataset',dataset,'--fold',str(fold)]})
        save_json(OUT/'jobs.json',jobs)
        summarize()
        print(f"已生成{len(jobs)}个五折任务",flush=True)
        return
    assert args.dataset
    exp,core,p=initialize(args.dataset)
    if args.initialize:return
    assert args.fold
    import torch
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.42)
    target=folder(args.dataset,args.fold)
    target.mkdir(parents=True,exist_ok=True)
    with (target/'exclusive.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (OUT/'completed'/f'{args.dataset}_fold_{args.fold}.json').exists():
            if args.dataset=='ct_rate':
                exp.core.worker(exp.arguments(ROOTS['ct_rate'],VARIANT,args.fold))
            else:
                (exp.base.run_amos if args.dataset=='amos_mm' else exp.base.run_mr)(VARIANT,args.fold)
            verify_fold(args.dataset,args.fold,p)
    summarize()


if __name__=='__main__':
    main()
