#!/usr/bin/env python3
"""核对CT适配位置机制与固定版本官方公式实现的数值及梯度。"""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import run_ctrate_680_all_models as core
import torch
from torch import nn
from torch.nn import functional as functional
from einops import rearrange
from exp_8.position_baselines import ComRoPEAP, VideoRoPE, DAPEV2Kerple, path_causal_scores, PositionSelfAttention

ROOT = core.ROOT
REFERENCES = ROOT / "src/vendor/position_references"
OUT = ROOT / "outputs/ct_rate_680/position_baselines"


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError(f"官方实现接口变化：{path}")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20260919)
    torch.backends.mha.set_fastpath_enabled(False)
    report = {}
    class AttentionDimensions(nn.Module):
        def __init__(self, configuration):
            super().__init__()
            self.config = configuration
            self.num_heads = configuration.num_attention_heads
            self.head_dim = configuration.hidden_size // self.num_heads
    environment = dict(torch=torch, nn=nn, CLIPAttention=AttentionDimensions,
                       CustomVITConfig=SimpleNamespace, Optional=Optional, Tuple=Tuple)
    source = REFERENCES / "comrope/model.py"
    definitions(source, ["RoPESelfAttentionBase", "ComRoPEAPAttention"], environment)
    config = SimpleNamespace(hidden_size=32, num_attention_heads=2, attention_dropout=0.0,
                             block_size=2, init_std=1.0, num_axes=1)
    reference = environment["ComRoPEAPAttention"](config)
    module = ComRoPEAP(2, 16)
    module.freqs.data.copy_(reference.freqs[:, 0])
    positions = torch.linspace(0, 1, 8)[None].expand(2, -1)
    reference.set_positions(positions[..., None])
    query = torch.randn(2, 2, 8, 16, requires_grad=True)
    key = torch.randn_like(query)
    actual, _ = module(query, key, positions)
    rotation = reference.get_rotation_matrix().transpose(1, 2)
    expected = (rotation @ query[..., None]).squeeze(-1)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    actual_grad = torch.autograd.grad(actual.square().sum() + actual.sum(), module.freqs)[0]
    expected_grad = torch.autograd.grad(expected.square().sum() + expected.sum(), reference.freqs)[0][:, 0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-5, rtol=2e-5)
    report["comrope_ap"] = {"official_rotation_max_error": (actual - expected).abs().max().item(),
                             "official_gradient_max_error": (actual_grad - expected_grad).abs().max().item()}
    environment = dict(torch=torch)
    definitions(REFERENCES / "videorope/videorope-transformer/modeling_videorope.py",
                ["rotate_half", "apply_m_modify_multimodal_rotary_pos_emb"], environment)
    module = VideoRoPE(128)
    coordinates = torch.randn(2, 3, 8)
    phase = coordinates[..., None] * module.inverse_frequency
    phase = torch.cat((phase, phase), -1).permute(1, 0, 2, 3)
    query, key = torch.randn(2, 4, 8, 128), torch.randn(2, 4, 8, 128)
    actual = module(query, key, coordinates)
    expected = environment["apply_m_modify_multimodal_rotary_pos_emb"](query, key, phase.cos(), phase.sin(), [16, 24, 24])
    for result, target in zip(actual, expected):
        torch.testing.assert_close(result, target, atol=0, rtol=0)
    report["videorope"] = {"official_multi_axis_exact_match": True,
                           "ct_1x1_grid_limitation": "三轴在对角线重合，退化为间距2的RoPE，不能检验完整空间机制"}
    environment = dict(torch=torch, nn=nn)
    definitions(REFERENCES / "dape_v2/megatron/model/positional_embeddings.py", ["Kerple_DAPEV2"], environment)
    arguments = SimpleNamespace(num_attention_heads=4, pos_emb="dapev2", noise_seq_length=64,
                                params_dtype=torch.float32, mlp_width=32, dapev2_kernel=3)
    reference = environment["Kerple_DAPEV2"](arguments, 1).cuda()
    module = DAPEV2Kerple(4).cuda()
    module.load_state_dict(reference.state_dict())
    scores = torch.randn(2, 4, 8, 8, device="cuda", requires_grad=True)
    actual = module(scores, torch.ones(2, 8, dtype=torch.bool, device="cuda"), causal=True)
    expected = reference(scores)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    actual_grad = torch.autograd.grad(actual.square().mean(), scores, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.square().mean(), scores)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=2e-6)
    report["dape_v2_kerple"] = {"official_causal_max_error": (actual - expected).abs().max().item(),
                                 "official_score_gradient_match": True,
                                 "ct_adaptation": "双向绝对相对距离与完整有效分数图；保留1×3卷积与残差"}
    environment = dict(torch=torch, F=functional, rearrange=rearrange)
    definitions(REFERENCES / "path/fla/ops/path_attn/naive.py", ["naive_path_attn"], environment)
    query = torch.randn(2, 2, 8, 16, dtype=torch.double, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query)
    direction = functional.normalize(torch.randn_like(query), dim=-1).requires_grad_()
    beta = torch.sigmoid(torch.randn(2, 2, 8, dtype=torch.double)).mul(2).requires_grad_()
    scores = path_causal_scores(query, key, direction, beta)
    actual = (scores / 4).masked_fill(torch.ones(8, 8, dtype=torch.bool).triu(1), -torch.inf).softmax(-1) @ value
    expected = environment["naive_path_attn"](
        query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), direction.transpose(1, 2),
        beta.transpose(1, 2), torch.zeros(2, 8, 2), scale=0.25, chunk_size=8).transpose(1, 2)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    reconstructed = []
    for query_index in range(8):
        transformed = key
        for step in range(1, query_index + 1):
            reflector = direction[:, :, step, None]
            updated = transformed - beta[:, :, step, None, None] * (transformed * reflector).sum(-1, keepdim=True) * reflector
            valid_update = torch.arange(8)[None, None, :, None] < step
            transformed = torch.where(valid_update, updated, transformed)
        reconstructed.append((query[:, :, query_index, None] * transformed).sum(-1))
    naive_scores = torch.stack(reconstructed, dim=-2).tril()
    torch.testing.assert_close(scores.tril(), naive_scores, atol=1e-10, rtol=1e-10)
    for actual_grad, expected_grad in zip(torch.autograd.grad(scores.tril().sum(), (query, key, direction, beta), retain_graph=True),
                                          torch.autograd.grad(naive_scores.sum(), (query, key, direction, beta))):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-9, rtol=1e-9)
    report["path"] = {"official_causal_output_max_error": (actual - expected).abs().max().item(),
                        "rank_one_recurrence_forward_and_gradients_match": True,
                        "ct_adaptation": "共享参数的正反两个因果核，拼接上下三角后统一softmax；非论文原始因果架构"}
    original = nn.MultiheadAttention(512, 4, dropout=0, batch_first=True).eval()
    identity = PositionSelfAttention(original, "identity").eval()
    features = torch.randn(2, 8, 512)
    reference = original(features, features, features, need_weights=False)[0]
    actual = identity(features, features, features, need_weights=False)[0]
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)
    report["attention_adapter_identity_max_error"] = (actual - reference).abs().max().item()
    for variant in ("comrope_ap", "videorope", "path", "dape_v2_kerple"):
        print(f"检查{variant}填充不变性", flush=True)
        attention = PositionSelfAttention(nn.MultiheadAttention(512, 4, dropout=0, batch_first=True), variant).eval()
        padded = torch.cat((features, torch.randn(2, 3, 512)), dim=1)
        padding = torch.zeros(2, 11, dtype=torch.bool)
        padding[:, -3:] = True
        expected = attention(features, features, features)[0]
        actual = attention(padded, padded, padded, key_padding_mask=padding)[0][:, :8]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        report[variant]["padding_invariance"] = True
    core.save_json(OUT / "implementation_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
