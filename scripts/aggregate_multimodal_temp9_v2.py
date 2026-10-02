#!/usr/bin/env python3
"""Aggregate the unified classification-attribution intervention outputs."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/lccf_public_evidence_20261001'
RAW = OUT / 'multimodal_interventions_v2' / 'raw'
DATASETS = {
    'ct_rate': ['Emph.', 'Atel.', 'Fibrotic'],
    'mr_rate': ['Uns.', 'Neuro.', 'Cerebro.', 'Neo.'],
    'amos_mm': ['L', 'K', 'G', 'S', 'B', 'P', 'T'],
    'aa_mini': ['Liver', 'Pancreas', 'Kidney'],
}
METHODS = ['mmfnet','radfuse','saif','mmtf','camchex','med3dvlm','m3fm','unified_mm','adaptive_fusion']


def collapse(case, values):
    return np.stack([values[case == c].mean(axis=0) for c in np.unique(case)])


def load(ds, condition):
    parts=[]
    for fold in range(1,6):
        with np.load(RAW/ds/condition/f'fold_{fold}.npz', allow_pickle=False) as z:
            parts.append({k:z[k] for k in z.files})
    return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}


def matrix(d, labels):
    n=len(labels); out=np.zeros((n,n),float)
    for source in range(n):
        valid=(d['image_count']>1)
        changes=np.abs(d['deletion_signed'][valid,source,:])
        means=collapse(d['case'][valid],changes).mean(0)
        out[source]=100*means/means.sum() if means.sum()>0 else np.nan
    return out


def main():
    rows=[]; overlaps={}
    for ds,labels in DATASETS.items():
        for condition in METHODS+['full','shared_control']:
            d=load(ds,condition)
            overlaps[(ds,condition)]=float(np.nanmean(d['overlap']))*100
            mat=matrix(d,labels)
            method='a_shared' if condition=='shared_control' else condition
            for source,sname in enumerate(labels):
                for target,tname in enumerate(labels):
                    rows.append({'dataset':ds,'method':method,'source':sname,'target':tname,'value':float(mat[source,target])})
    out=OUT/'multimodal_intervention_summary_v2.csv'
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['dataset','method','source','target','value']);w.writeheader();w.writerows(rows)
    with (OUT/'multimodal_overlap_v2.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['dataset','method','value']);w.writeheader()
        for (ds,m),v in overlaps.items():w.writerow({'dataset':ds,'method':('a_shared' if m=='shared_control' else ('full' if m=='full' else m)),'value':v})
    print(out)


if __name__=='__main__':main()
