#!/usr/bin/env python3
import os, json
repair = "/xmlg/Lim/Project4/outputs/amos_mm/repair_amos_5964/amos_5964.nii.gz"
current = "/xmlg/Lim/Project4/datasets/amos_mm/imagesTr/amos_5964.nii.gz"
r_size = os.path.getsize(repair)
c_size = os.path.getsize(current)
print(f"Repair file: {r_size} bytes")
print(f"Current file: {c_size} bytes")
print(f"Repair is larger: {r_size > c_size}")

status_path = "/xmlg/Lim/Project4/outputs/amos_mm/experiment/feature_status.json"
with open(status_path) as f:
    status = json.load(f)
print(f"Feature status: {json.dumps(status, indent=2)}")

missing_feature = None
features_dir = "/xmlg/Lim/Project4/outputs/amos_mm/experiment/features"
samples_path = "/xmlg/Lim/Project4/outputs/amos_mm/experiment/samples.json"
with open(samples_path) as f:
    samples = json.load(f)
for s in samples:
    if s["scan_id"] == "amos_5964":
        idx = s["case_index"]
        feat_path = os.path.join(features_dir, f"{idx:04d}.npz")
        missing_feature = feat_path
        print(f"amos_5964 case_index={idx}, feature exists={os.path.exists(feat_path)}")
        break