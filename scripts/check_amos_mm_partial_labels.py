#!/usr/bin/env python3
"""回归检查：全标签监督与原模型等价，未知标签填充值不改变监督损失。"""
import types

import numpy as np
import torch

import run_amos_mm_all_models as run
from amos_mm_model_adapters import forward_image, forward_fusion, masked_asl
from exp_8.models import Exp12AProCoPEWatchCrossAttentionTextCNNModel


def check():
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    results = []
    images = torch.randn(2, 8, 768, 1, 1, device="cuda")
    image_mask = torch.ones(2, 8, dtype=torch.bool, device="cuda")
    token_ids = torch.randint(1, 100, (2, 512), device="cuda")
    token_mask = torch.ones_like(token_ids, dtype=torch.bool)
    positions = torch.arange(8, device="cuda").expand(2, -1) * 3
    counts = torch.tensor([25, 25], device="cuda")
    for c in (3, 7):
        y = torch.tensor([[1, 0, 0, 1, 0, 1, 0], [0, 1, 0, 0, 1, 0, 1]], dtype=torch.float32, device="cuda")[:, :c]
        all_known = torch.ones_like(y, dtype=torch.bool)
        for key in ("clam_mb", "clam_sb", "dtfd_mil", "task2_mmfnet_2024", "amef_multimodal"):
            model, weights = run.build_model(key, c)
            model.cuda().train()

            def adapted(target, known):
                torch.manual_seed(99)
                if key in run.IMAGE_MODELS:
                    return forward_image(model, key, images, image_mask, target, known, True)
                return forward_fusion(model, key, images, image_mask, token_ids, token_mask,
                                      positions, counts, target, known, True)

            with torch.no_grad():
                new = adapted(y, all_known)
                if not (key == "amef_multimodal" and c == 7):
                    torch.manual_seed(99)
                    if key in run.IMAGE_MODELS:
                        old = model(images, image_mask, labels=y)
                    else:
                        if key == "amef_multimodal":
                            model._build_watch_cross_attention_outputs = types.MethodType(
                                Exp12AProCoPEWatchCrossAttentionTextCNNModel._build_watch_cross_attention_outputs, model)
                        old = model(images, image_mask, labels=y, watch_token_ids=token_ids, watch_token_mask=token_mask,
                                    instance_indices=positions, original_image_counts=counts)
                        if key == "amef_multimodal":
                            del model._build_watch_cross_attention_outputs
                    assert torch.allclose(new["logits"], old["logits"], atol=1e-6, rtol=1e-5), (key, c, "logits")
                    for name, weight in weights.items():
                        if weight:
                            assert torch.allclose(new["aux_losses"][name], old["aux_losses"][name], atol=1e-6, rtol=1e-5), (key, c, name)
                known = all_known.clone()
                known[:, 2] = False
                changed = y.clone()
                changed[:, 2] = 1 - changed[:, 2]
                a, b = adapted(y, known), adapted(changed, known)
                assert torch.allclose(masked_asl(a["logits"], y, known), masked_asl(b["logits"], changed, known))
                for name, weight in weights.items():
                    if weight:
                        assert torch.allclose(a["aux_losses"][name], b["aux_losses"][name]), (key, c, name, "unknown")
                results.append({"labels": c, "model": key, "unknown_value_invariant": True,
                                "all_known_matches_original": None if key == "amef_multimodal" and c == 7 else True})
                print(results[-1], flush=True)
            del model, new, a, b
            torch.cuda.empty_cache()
    run.save_json(run.OUT / "partial_label_regression_checks.json", results)


if __name__ == "__main__":
    check()
