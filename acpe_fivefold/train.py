"""完整训练一个开发折，保存断点，仅评估开发验证集。"""
import argparse
import fcntl
import json
import math
import os
import random
import signal
import time
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from .common import ROOT, OUT, BASE, CONDITIONS, folder, read, write, digest, identity, key
from .model import Model
from auto_research.train import Bags, collate, cuda_batch, evaluate, loss_fn, label_f1, gradient_diagnostic
from training.data import _encode_text_fields


def save(path, obj):
    temp = path.with_suffix('.tmp')
    torch.save(obj, temp)
    temp.replace(path)


class FullBags(Dataset):
    def __init__(self, rows, cache, full, ids, condition):
        self.rows, self.cache, self.full, self.ids, self.condition = rows, cache, full, ids, condition

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        case = self.ids[index]
        item = self.full[case]
        selected, local = item['selections'][self.condition]
        row = self.rows[case]
        return (item['features'][local], selected, item['count'], *self.cache[case][3:],
                np.asarray(row['labels'],np.float32),
                np.asarray(row.get('known_mask',[True]*len(row['labels'])),bool), case)


class StructuredBags(Bags):
    """训练病例使用预登记的原序列删片视图；相同增强同时用于匹配对照。"""
    def __init__(self,*args,full,**kwargs):
        super().__init__(*args,**kwargs)
        self.full=full

    def __getitem__(self,index):
        case=self.ids[index]
        if case not in self.full or np.random.random()>=.5:
            return super().__getitem__(index)
        conditions=[key(r,b) for r,b in ((40,1),(60,1),(80,1),(80,4),(80,8))]
        condition=conditions[int(np.random.randint(len(conditions)))]
        return FullBags(self.rows,self.cache,self.full,[case],condition)[0]


def load_full(ds,ids):
    manifest=read(OUT/'deletion_manifest.json')[ds]
    full={};hashes={}
    for case in tqdm(ids,desc='读取原序列删除视图',mininterval=10):
        info=manifest[str(case)]
        if not info['eligible']:continue
        path=OUT/'features'/ds/f'{case:04d}.npz'
        with np.load(path,allow_pickle=False) as z:
            ix=z['source_indices'].astype(np.int64);features=z['features'].astype(np.float32)
            assert int(z['original_count'])==info['source_count']
        selections={}
        for condition,selected in info['selections'].items():
            selected=np.asarray(selected,dtype=np.int64);local=np.searchsorted(ix,selected)
            assert np.array_equal(ix[local],selected)
            selections[condition]=(selected,local)
        full[case]={'features':features,'selections':selections,'count':info['source_count']}
        hashes[str(case)]=digest(path)
    return full,hashes


def metric(value, threshold=.5):
    return {'f1': float(label_f1(value['y'],value['p'],value['known'],threshold).mean()),
            'loss': float(value['loss']), 'n': len(value['ids'])}


def run(spec):
    plan = read(OUT/'protocol.json')
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    assert spec['protocol_sha256'] == plan['sha256']
    ds, fold, config = spec['dataset'], spec['fold'], spec['config']
    outdir = OUT/'runs'/spec['id']
    outdir.mkdir(parents=True, exist_ok=True)
    lock = (outdir/'run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (outdir/'result.json').exists():
        assert read(outdir/'result.json')['spec_sha256'] == identity(spec)
        return
    for name, expected in plan['baseline_source_hashes'].items():
        assert digest(BASE/name) == expected, name
    for name, expected in plan['source_hashes'].items():
        assert digest(ROOT/'src/acpe_fivefold'/name) == expected, name
    split = plan['datasets'][ds]['folds'][fold-1]
    assert split['fold'] == fold
    seed = split['seed']
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cudnn.benchmark = False
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    rows = {r['case_index']:r for r in read(folder(ds)/'samples.json')}
    for name, expected in plan['datasets'][ds]['input_hashes'].items():
        assert digest(folder(ds)/name) == expected
    ids = sorted(split['train']+split['validation'])
    assert not set(ids) & set(plan['datasets'][ds]['excluded_test'])
    cache, hashes = {}, {}
    for case in tqdm(ids,desc='读取开发特征',mininterval=10):
        path = folder(ds)/'features'/f'{case:04d}.npz'
        with np.load(path,allow_pickle=False) as z:
            values=z['features'].astype(np.float32)
            positions=z['slice_indices'].astype(np.int64)
            count=int(z['original_count'])
        assert values.shape==(len(positions),768) and np.isfinite(values).all()
        tokens = _encode_text_fields({'watch':rows[case]['findings_masked']},('watch',),max_length=512,vocab_size=8192)
        cache[case]=(values,positions,count,*tokens)
        hashes[str(case)]=digest(path)
    write(outdir/'protocol.json',{'spec':spec,'split':split,'excluded_test':plan['datasets'][ds]['excluded_test'],
                                 'torch':torch.__version__,'numpy':np.__version__})
    write(outdir/'cache_hashes.json',hashes)
    model=Model(len(rows[ids[0]]['labels']),config).cuda()
    torch.manual_seed(seed+1000);torch.cuda.manual_seed_all(seed+1000)
    lr=float(config.get('lr',plan['lr']))
    groups=[{'params':[p for n,p in model.named_parameters() if 'apro_positioner' not in n],'lr':lr},
            {'params':[p for n,p in model.named_parameters() if 'apro_positioner' in n],
             'lr':lr*float(config.get('position_lr_multiplier',1))}]
    optimizer=torch.optim.AdamW(groups,weight_decay=plan['weight_decay'])
    full,feature_hashes=load_full(ds,ids if config.get('structured_training') else split['validation'])
    def loader(indices,training=False):
        bags=(StructuredBags(rows,indices,cache,config,training=True,full=full)
              if training and config.get('structured_training') else Bags(rows,indices,cache,config,training=training))
        return DataLoader(bags,batch_size=plan['batch_size'],
                          shuffle=training,num_workers=0,collate_fn=collate)
    training,validation=loader(split['train'],True),loader(split['validation'])
    steps=plan['epochs']*len(training);warmup=max(1,int(.2*steps))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:
        (step+1)/warmup if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,steps-warmup))))
    scaler=torch.amp.GradScaler('cuda')
    best,best_epoch,history,start=float('inf'),0,[],0
    if (outdir/'last.pt').exists():
        ckpt=torch.load(outdir/'last.pt',map_location='cpu',weights_only=False)
        assert ckpt['spec_sha256']==identity(spec)
        model.load_state_dict(ckpt['model']);optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler']);scaler.load_state_dict(ckpt['scaler'])
        best,best_epoch,history,start=ckpt['best'],ckpt['best_epoch'],ckpt['history'],ckpt['epoch']
        random.setstate(ckpt['python_rng']);np.random.set_state(ckpt['numpy_rng'])
        torch.set_rng_state(ckpt['torch_rng']);torch.cuda.set_rng_state(ckpt['cuda_rng'])
        del ckpt
    t0=time.monotonic();previous=history[-1]['seconds'] if history else 0
    for epoch in range(start,plan['epochs']):
        model.train();model.epoch=epoch
        sums={'main':0.,'lqd':0.,'image':0.,'total':0.};diagnostic=None
        for step,batch in enumerate(training):
            inputs,y,known=cuda_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                output=model(**inputs,labels=(y,known))
            main=loss_fn(output['logits'],y,known)
            image=loss_fn(output['image_only_logits'],y,known)
            lqd=output['research_lqd']
            total=main+plan['lqd_weight']*lqd+float(config.get('image_weight',0))*image
            if config.get('coordinate_penalty'):
                difference=output['apro_context_coordinates']-output['apro_raw_coordinates']
                total=total+config['coordinate_penalty']*difference[inputs['mask']].square().mean()
            if step==0 and epoch in (0,4,9,19,29):
                diagnostic=gradient_diagnostic(model,{'main':main,'image':image,'lqd':lqd})
            if not torch.isfinite(total):raise FloatingPointError('训练损失非有限')
            scaler.scale(total).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer);scaler.update();scheduler.step()
            for k,v in [('main',main),('lqd',lqd),('image',image),('total',total)]:sums[k]+=float(v.detach())
        val=evaluate(model,validation)
        if not np.isfinite(val['p']).all() or not math.isfinite(val['loss']):raise FloatingPointError('验证输出非有限')
        if val['loss']<best:
            best,best_epoch=val['loss'],epoch+1
            save(outdir/'best.pt',{'model':model.state_dict(),'epoch':best_epoch,'config':config,
                                  'spec_sha256':identity(spec),'protocol_sha256':plan['sha256']})
        row={'epoch':epoch+1,'train':{k:v/len(training) for k,v in sums.items()},
             'validation':metric(val),'gradient':diagnostic,'seconds':previous+time.monotonic()-t0}
        history.append(row);write(outdir/'history.json',history)
        save(outdir/'last.pt',{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
             'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'best':best,'best_epoch':best_epoch,
             'history':history,'epoch':epoch+1,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
             'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'spec_sha256':identity(spec)})
        write(outdir/'status.json',{'phase':'training','epoch':epoch+1,'pid':os.getpid(),'updated':time.time()})
        print(json.dumps({'job':spec['id'],**row},ensure_ascii=False),flush=True)
        if stop_requested:
            write(outdir/'status.json',{'phase':'saved_pause','epoch':epoch+1,'pid':os.getpid(),'updated':time.time()})
            raise SystemExit(75)
    ckpt=torch.load(outdir/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(ckpt['model']);model.epoch=ckpt['epoch']-1
    clean=evaluate(model,validation)
    grid=np.round(np.arange(.1,.901,.05),2)
    thresholds=grid[np.stack([label_f1(clean['y'],clean['p'],clean['known'],t) for t in grid]).argmax(0)]
    for j in range(len(thresholds)):
        if len(np.unique(clean['y'][clean['known'][:,j],j]))<2:thresholds[j]=.5
    eligible=sorted(set(full)&set(split['validation']))
    assert eligible and not set(eligible)&set(plan['datasets'][ds]['excluded_test'])
    conditions={'clean':{**metric(clean),'tuned_f1':metric(clean,thresholds)['f1']}}
    predictions={f'clean_{k}':clean[k] for k in ('y','p','known','ids')}
    for ratio,blocks in CONDITIONS:
        condition=key(ratio,blocks)
        loader_full=DataLoader(FullBags(rows,cache,full,eligible,condition),batch_size=plan['batch_size'],
                               num_workers=0,collate_fn=collate)
        val=evaluate(model,loader_full)
        if not np.isfinite(val['p']).all():raise FloatingPointError('删除评估输出非有限')
        conditions[condition]={**metric(val),'tuned_f1':metric(val,thresholds)['f1'],
                               'anchor_agreement':val['anchor_agreement']}
        predictions.update({f'{condition}_{k}':val[k] for k in ('y','p','known','ids')})
    gains={name:p.detach().float().cpu().tolist() for name,p in model.named_parameters()
           if name.endswith(('absolute_gain','relative_gain'))}
    write(outdir/'deletion_feature_hashes.json',feature_hashes)
    np.savez_compressed(outdir/'validation_predictions.npz',**predictions,thresholds=thresholds)
    result={'status':'completed','spec_sha256':identity(spec),'protocol_sha256':plan['sha256'],
            'dataset':ds,'fold':fold,'seed':seed,'config':config,'conditions':conditions,
            'best_epoch':best_epoch,'thresholds':thresholds.tolist(),'eligible_ids':eligible,
            'best_sha256':digest(outdir/'best.pt'),'learned_gains':gains,'seconds':previous+time.monotonic()-t0,
            'evaluation_split':'development_validation','test_evaluated':False}
    write(outdir/'result.json',result)
    write(outdir/'status.json',{'phase':'completed','pid':os.getpid(),'updated':time.time()})
    # 已完成任务保留最佳权重和完整日志，删除本轮可重建的优化器断点以控制磁盘占用。
    (outdir/'last.pt').unlink(missing_ok=True)
    print('完成 '+spec['id'],flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True)
    args=parser.parse_args();spec=read(args.job)
    try:run(spec)
    except Exception:
        write(OUT/'runs'/spec['id']/'failure.json',{'error':traceback.format_exc(),'time':time.time(),'pid':os.getpid()})
        raise


if __name__=='__main__':main()
