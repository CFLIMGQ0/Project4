#!/usr/bin/env python3
import json
samples = json.load(open("/xmlg/Lim/Project4/outputs/amos_mm/experiment/samples.json"))
splits = json.load(open("/xmlg/Lim/Project4/outputs/amos_mm/experiment/splits.json"))
for s in samples:
    if s["scan_id"] == "amos_5964":
        idx = s["case_index"]
        print(f"amos_5964: case_index={idx}")
        break
dev = splits["development"]
test = splits["test"]
val_folds = splits["validation_folds"]
print(f"In development: {idx in dev}")
print(f"In test: {idx in test}")
for i, vf in enumerate(val_folds):
    if idx in vf:
        print(f"In validation fold {i}")
        break
import os
feat_path = f"/xmlg/Lim/Project4/outputs/amos_mm/experiment/features/{idx:04d}.npz"
print(f"Feature file exists: {os.path.exists(feat_path)}")