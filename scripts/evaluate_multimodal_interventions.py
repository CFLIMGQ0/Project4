#!/usr/bin/env python3
"""temp9 多模态比较：统一使用分类 logit 的梯度×输入选图，再删除并重放。

所有模型以相同的缓存图像特征为输入；报告不变，不更新参数。
Shared 是 full 的诊断对照：先对标签贡献取均值，所有标签删除同一组图像。
逐病例保存贡献得分、原始输出、删除后的输出和检查索引，支持结果核验。
"""
from __future__ import annotations
import argparse, gc, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from scripts import evaluate_lccf_public_figures as base
VERSION = 'logit_grad_x_input_compact_deletion_v2'
VARIANTS = (*base.MULTIMODAL, 'full')


def forward(model, x, mask, pos, counts, token, tm):
    return model(x, mask, instance_indices=pos, original_image_counts=counts,
                 watch_token_ids=token, watch_token_mask=tm)['logits']


def remove_selected(x, mask, pos, selected):
    # 删除后保持顺序并压紧有效槽位；补齐槽位只出现在序列末尾。
    kept = mask.clone()
    for bi in range(len(mask)):
        valid = selected[bi][selected[bi] >= 0]
        kept[bi, valid] = False
    length = int(kept.sum(1).max())
    xx = torch.zeros_like(x[:, :length]); pp = torch.full_like(pos[:, :length], -1)
    mm = torch.zeros_like(mask[:, :length])
    for bi in range(len(mask)):
        n = int(kept[bi].sum()); xx[bi, :n] = x[bi, kept[bi]]
        pp[bi, :n] = pos[bi, kept[bi]]; mm[bi, :n] = True
    return xx, mm, pp


def select_top(scores, mask):
    result = torch.full((*scores.shape[:2], 5), -1, device=scores.device, dtype=torch.long)
    for bi in range(len(mask)):
        k = min(5, int(mask[bi].sum()) - 1)
        if k > 0:
            result[bi, :, :k] = scores[bi].masked_fill(~mask[bi], -torch.inf).argsort(
                dim=-1, descending=True, stable=True)[:, :k]
    return result


def overlaps(indices):
    values = []
    for case in indices.cpu().numpy():
        sets = [set(a[a >= 0].tolist()) for a in case]
        values.append(np.mean([len(a & b)/len(a | b) if a | b else np.nan
                              for i,a in enumerate(sets) for b in sets[i+1:]]))
    return np.array(values)


def evaluate(ds, variant, fold, inputs, device, batch_size, out_root):
    rows, all_y, known, tokens, text_masks, bags = inputs
    folder = base.folder_for(ds, variant, fold)
    ids, ref_y, ref_p, ref_file = base.reference(folder, ds, rows)
    common_ids, _, _, _ = base.reference(base.folder_for(ds, 'full', fold), ds, rows)
    assert np.array_equal(ids, common_ids) and np.array_equal(all_y[ids], ref_y)
    assert known[ids].all()
    model, digest, config = base.load_model(folder, ds, variant, device)
    model.requires_grad_(False)
    nlabels = base.DATASETS[ds][2]
    conditions = [variant] + (['shared_control'] if variant == 'full' else [])
    gathered = {c: {} for c in conditions}
    amp = ds == 'amos_mm' and variant != 'full'
    started = time.monotonic()
    def add(condition,key,value):
        if torch.is_tensor(value):value=value.detach().float().cpu().numpy()
        gathered[condition].setdefault(key,[]).append(value)
    for start in range(0,len(ids),batch_size):
        case = ids[start:start+batch_size]
        x, mask, pos, counts, token, tm = base.pack(case,bags,tokens,text_masks,device)
        # 贡献按FP32的真实分类路径计算，避免半精度梯度下溢；不使用未参与分类的注意力。
        x.requires_grad_(True)
        with torch.enable_grad():
            logits = forward(model,x,mask,pos,counts,token,tm)
            scores=[]
            for label in range(nlabels):
                grad, = torch.autograd.grad(logits[:,label].sum(),x,
                                             retain_graph=label<nlabels-1)
                scores.append((grad*x).sum((2,3,4)).detach())
        scores=torch.stack(scores,1)
        assert torch.isfinite(scores).all()
        x=x.detach()
        with torch.no_grad(), torch.autocast(device.type,enabled=amp):
            original=forward(model,x,mask,pos,counts,token,tm)
            # 历史AMOS记录使用FP16 logits上sigmoid，遵循相同路径进行复核。
            prob=original.sigmoid().float()
        for condition in conditions:
            sc=scores if condition!='shared_control' else scores.mean(1,keepdim=True).expand_as(scores)
            top=select_top(sc,mask)
            changed=[]
            with torch.no_grad(),torch.autocast(device.type,enabled=amp):
                for source in range(nlabels):
                    xx,mm,pp=remove_selected(x,mask,pos,top[:,source])
                    p=forward(model,xx,mm,pp,counts,token,tm).sigmoid().float()
                    changed.append(p)
            changed=torch.stack(changed,1)
            delta=prob[:,None]-changed
            delta[mask.sum(1)<2]=torch.nan
            add(condition,'prob',prob);add(condition,'deletion_prob',changed)
            add(condition,'deletion_signed',delta);add(condition,'overlap',overlaps(top))
            add(condition,'image_count',mask.sum(1));add(condition,'top_indices',top)
            # 输入最多64张，统一补齐后保存，便于逐病例复现Top5选择。
            pad=torch.full((len(case),nlabels,64),torch.nan,device=device)
            pad[:,:,:sc.shape[-1]]=sc
            add(condition,'scores',pad)
        del logits,grad,scores
    replay=np.concatenate(gathered[variant]['prob']); error=np.abs(replay-ref_p)
    tolerance=.005 if variant in base.MULTIMODAL else 3e-5
    passed=bool(np.isfinite(replay).all() and error.max()<=tolerance)
    meta={
        'protocol_version':VERSION,'dataset':ds,'variant':variant,'fold':fold,
        'checkpoint':str((folder/'best_model.pt').relative_to(ROOT)),
        'checkpoint_mtime_ns':(folder/'best_model.pt').stat().st_mtime_ns,
        'training_protocol_sha256':digest,'cases':len(ids),'unique_cases':len(np.unique(ids)),
        'reference_predictions':str((folder/ref_file).relative_to(ROOT)),
        'probability_max_abs_error':float(error.max()),'probability_mean_abs_error':float(error.mean()),
        'tolerance':tolerance,'test_reference_passed':passed,'batch_size':batch_size,
        'prediction_precision':'AMP FP16' if amp else 'FP32','attribution_precision':'FP32',
        'elapsed_seconds':time.monotonic()-started,
        'attribution':'sum_feature(input * d(label_logit)/d(input)); signed scores ranked descending',
        'intervention':'delete top5, retain at least one image; compact sequence; preserve original acquisition indices; report fixed; rerun full prediction',
    }
    for condition in conditions:
        target=out_root/'raw'/ds/condition;target.mkdir(parents=True,exist_ok=True)
        data={k:np.concatenate(v) for k,v in gathered[condition].items()}
        data.update(case=ids,fold=np.full(len(ids),fold),labels=ref_y,reference_prob=ref_p)
        np.savez_compressed(target/f'fold_{fold}.npz',**data)
        entry=dict(meta,variant=condition)
        (target/f'fold_{fold}.json').write_text(json.dumps(entry,ensure_ascii=False,indent=2))
    del model;gc.collect();torch.cuda.empty_cache()
    if not passed:raise RuntimeError(f'{ds}/{variant}/fold{fold}: 预测复核失败 {error.max()}')
    return meta


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--datasets',nargs='+',choices=base.DATASETS,default=list(base.DATASETS))
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    p.add_argument('--folds',nargs='+',type=int,default=[1,2,3,4,5])
    p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--out',default=str(base.OUT/'multimodal_interventions_v2'))
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(2026);torch.backends.mha.set_fastpath_enabled(False)
    device=torch.device(args.device);out=Path(args.out)
    for ds in args.datasets:
        inputs=base.load_inputs(ds)
        for variant in args.variants:
            for fold in tqdm(args.folds,desc=f'{ds}/{variant}',mininterval=10):
                target=out/'raw'/ds/variant/f'fold_{fold}.json'
                if args.resume and target.exists():
                    m=json.loads(target.read_text())
                    if m.get('protocol_version')==VERSION and m.get('test_reference_passed'):continue
                result=evaluate(ds,variant,fold,inputs,device,args.batch_size,out)
                print(f"{ds}/{variant}/fold{fold}: 完成 {result['elapsed_seconds']:.1f}s, 复核 {result['probability_max_abs_error']:.2g}",flush=True)
if __name__=='__main__':main()
