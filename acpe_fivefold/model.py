"""不改历史实现的ACPE机制候选；所有位置变形保留顺序和端点。"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .common import BASE
from auto_research.model import ResearchModel, ResearchPosition


class Position(ResearchPosition):
    def _contextual_coordinates(self, features, mask, raw):
        cfg = self.research_config
        if cfg.get('position') == 'anchored':
            return raw, torch.zeros_like(raw)
        custom = any(cfg.get(k) for k in ('compact_descriptor', 'context_form', 'eta_confidence', 'gap_confidence'))
        if not custom:
            return super()._contextual_coordinates(features, mask, raw)
        u = self.transition_projector(features) * mask[..., None]
        form = cfg.get('context_form')
        if form == 'smooth':
            u = (F.avg_pool1d(u.transpose(1, 2), 3, 1, 1).transpose(1, 2) /
                 F.avg_pool1d(mask[:, None].float(), 3, 1, 1).transpose(1, 2).clamp_min(1e-6))
        pair = mask[:, 1:] & mask[:, :-1]
        gap = torch.zeros_like(raw)
        gap[:, 1:] = (raw[:, 1:] - raw[:, :-1]).clamp_min(1e-6) * pair
        delta = (u[:, 1:] - u[:, :-1]) * pair[..., None]
        if form == 'velocity':
            mean_gap = gap.sum(1, keepdim=True) / pair.sum(1, keepdim=True).clamp_min(1)
            relative_gap = (gap[:, 1:] / mean_gap.clamp_min(1e-6)).clamp(.25, 4.)
            delta = delta / relative_gap[..., None]
        back, ahead = torch.zeros_like(u), torch.zeros_like(u)
        back[:, 1:], ahead[:, :-1] = delta, delta
        current = torch.zeros_like(u) if cfg.get('descriptor') == 'no_current' else u
        if form == 'global':
            current = (u.sum(1, keepdim=True) / mask.sum(1)[:, None, None].clamp_min(1)).expand_as(u) * mask[..., None]
        parts = [current, back, ahead, back.abs(), ahead.abs(), back * ahead]
        if form == 'magnitude':
            parts = [torch.zeros_like(x) if j in (0, 1, 2, 5) else x for j, x in enumerate(parts)]
        elif form == 'direction':
            parts = [torch.zeros_like(x) if j in (0, 3, 4, 5) else x for j, x in enumerate(parts)]
        if cfg.get('compact_descriptor'):
            parts = parts[1:]
        eta = self.warp_alpha * torch.tanh(self.transition_mlp(torch.cat(parts + [gap[..., None]], -1)).squeeze(-1))
        if cfg.get('eta_confidence'):
            both = torch.zeros_like(mask)
            both[:, 1:-1] = mask[:, :-2] & mask[:, 1:-1] & mask[:, 2:]
            agreement = .5 * (1 + F.cosine_similarity(back.float(), ahead.float(), dim=-1))
            confidence = torch.where(both, agreement, torch.ones_like(agreement))
            eta = eta * confidence.to(eta.dtype)
        if cfg.get('gap_confidence'):
            mean_gap = gap.sum(1, keepdim=True) / pair.sum(1, keepdim=True).clamp_min(1)
            eta = eta * (mean_gap / gap.clamp_min(1e-6)).clamp_max(1)
        eta = eta * torch.cat([torch.zeros_like(mask[:, :1]), pair], 1)
        weight = gap * eta.float().exp().to(gap.dtype)
        end = raw.gather(1, (mask.sum(1)-1).clamp_min(0)[:, None])
        adjusted = weight * (end-raw[:, :1]) / weight.sum(1, keepdim=True).clamp_min(1e-8)
        contextual = (raw[:, :1] + adjusted.cumsum(1)) * mask
        return contextual, eta

    def forward(self, features, mask, indices, counts):
        absolute, bias, extra = super().forward(features, mask, indices, counts)
        if self.research_config.get('correction_only'):
            raw = extra['apro_raw_coordinates']
            zero = torch.zeros_like(raw)
            if absolute is not None:
                anchored = self.absolute_projector(torch.cat([self.fourier(raw), self.fourier(raw), self.fourier(zero)], -1))
                absolute = absolute - anchored * mask[..., None]
            if bias is not None:
                distance = raw[:, :, None] - raw[:, None, :]
                anchored = 2 * torch.tanh(self.relative_bias(torch.cat([
                    self.fourier(distance), self.fourier(distance), self.fourier(torch.zeros_like(distance))], -1)))
                valid = mask[:, None, :, None] & mask[:, None, None, :]
                bias = bias - anchored.permute(0, 3, 1, 2) * valid
        if hasattr(self, 'absolute_gain'):
            if absolute is not None:
                absolute = absolute * self.absolute_gain.to(absolute.dtype)
            if bias is not None:
                bias = bias * self.relative_gain[None, :, None, None].to(bias.dtype)
        if hasattr(self, 'content_gate'):
            summary = (features * mask[..., None]).sum(1) / mask.sum(1)[:, None].clamp_min(1)
            gates = torch.sigmoid(self.content_gate(summary))
            if absolute is not None:
                absolute = absolute * gates[:, :1, None]
            if bias is not None:
                bias = bias * gates[:, 1:, None, None]
        return absolute, bias, extra


class Model(ResearchModel):
    def __init__(self, num_labels, config):
        super().__init__(num_labels, config)
        p = self.apro_positioner
        p.__class__ = Position
        if config.get('compact_descriptor'):
            dim = 5 * p.transition_dim + 1
            p.transition_mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, p.transition_dim),
                                             nn.GELU(), nn.Dropout(.2), nn.Linear(p.transition_dim, 1))
            if config.get('zero_warp_init'):
                nn.init.zeros_(p.transition_mlp[-1].weight)
                nn.init.zeros_(p.transition_mlp[-1].bias)
        if 'learned_scale' in config:
            p.absolute_gain = nn.Parameter(torch.full((p.feature_dim,), float(config['learned_scale'])))
            p.relative_gain = nn.Parameter(torch.full((p.num_heads,), float(config['learned_scale'])))
        if config.get('content_gate'):
            p.content_gate = nn.Linear(p.feature_dim, 1 + p.num_heads)
            nn.init.zeros_(p.content_gate.weight)
            nn.init.constant_(p.content_gate.bias, math.log(.1 / .9))

    def _encode_position(self, features, mask, indices, counts):
        positioned, bias, extra = super()._encode_position(features, mask, indices, counts)
        if self.config.get('base_original'):
            original = self.position_encoding(features, mask)
            positioned = positioned + original - features * mask[..., None]
        return positioned, bias, extra
