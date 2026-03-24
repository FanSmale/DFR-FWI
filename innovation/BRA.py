# -*- coding: utf-8 -*-
"""
@Time : 2025/11/21
@Author : Zeng Zifei (modified for DL-FWI with NCHW support)

Bi-Level Routing Attention (BRA) module
Input/Output format: (B, C, H, W) — standard PyTorch NCHW
Internal computation: converted to NHWC for region partitioning
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


class TopkRouting(nn.Module):
    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor, Tensor]:
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key)

        attn_logit = (query_hat * self.scale) @ key_hat.transpose(-2, -1)

        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)

        r_weight = self.routing_act(topk_attn_logit)
        return r_weight, topk_index


class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):

        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx: Tensor, r_weight: Tensor, kv: Tensor):
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)

        topk_kv = torch.gather(
            kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),
            dim=2,
            index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv)
        )

        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv
        return topk_kv


class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)

    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim + self.dim], dim=-1)
        return q, kv


class BiLevelRoutingAttention(nn.Module):
    """
    Bi-Level Routing Attention for DL-FWI.

    Input:  (B, C, H, W) — standard PyTorch format
    Output: (B, C, H, W)

    Args:
        dim: input/output channel dimension
        num_heads: number of attention heads (must divide dim and qk_dim)
        n_win: number of windows per side (S in paper), total regions = n_win * n_win
        ... (other args same as original)
    """

    def __init__(self, dim, num_heads=8, n_win=7, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None,
                 kv_downsample_mode='identity', topk=4, param_attention="qkvo",
                 param_routing=False, diff_routing=False, soft_routing=False,
                 side_dwconv=3, auto_pad=False):
        super().__init__()
        self.dim = dim
        # print(f"[BRA INIT] dim={dim}, num_heads={num_heads}, n_win={n_win}")

        self.n_win = n_win
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        self.auto_pad = auto_pad

        assert self.qk_dim % num_heads == 0 and self.dim % num_heads == 0, \
            f'qk_dim ({self.qk_dim}) and dim ({self.dim}) must be divisible by num_heads ({num_heads})!'
        self.scale = qk_scale or self.qk_dim ** -0.5

        # Local Context Enhancement (LCE)
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1,
                              padding=side_dwconv // 2, groups=dim) if side_dwconv > 0 else \
            lambda x: torch.zeros_like(x)

        # Routing
        assert not (param_routing and not diff_routing), "param_routing requires diff_routing=True"
        self.router = TopkRouting(
            qk_dim=self.qk_dim,
            qk_scale=self.scale,
            topk=topk,
            diff_routing=diff_routing,
            param_routing=param_routing
        )
        mul_weight = 'soft' if soft_routing else ('hard' if diff_routing else 'none')
        self.kv_gather = KVGather(mul_weight=mul_weight)

        # QKV & Output projection
        if param_attention == 'qkvo':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif param_attention == 'qkv':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else:
            raise ValueError(f'param_attention mode {param_attention} not supported!')

        self.kv_downsample_mode = kv_downsample_mode
        if kv_downsample_mode == 'ada_avgpool':
            assert kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2d(kv_per_win)
        elif kv_downsample_mode == 'ada_maxpool':
            assert kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2d(kv_per_win)
        elif kv_downsample_mode == 'maxpool':
            self.kv_down = nn.MaxPool2d(kv_downsample_ratio) if kv_downsample_ratio > 1 else nn.Identity()
        elif kv_downsample_mode == 'avgpool':
            self.kv_down = nn.AvgPool2d(kv_downsample_ratio) if kv_downsample_ratio > 1 else nn.Identity()
        elif kv_downsample_mode == 'identity':
            self.kv_down = nn.Identity()
        else:
            raise ValueError(f'kv_downsample_mode {kv_downsample_mode} not supported!')

        self.attn_act = nn.Softmax(dim=-1)

    def forward(self, x, ret_attn_mask=False):
        """
        Input:  x - (B, C, H, W)
        Output: (B, C, H, W)
        """
        # Save original shape for padding restore
        B, H_in, W_in, C = x.shape
        # print(f"[BRA FORWARD] Input NCHW: ({B}, {C}, {H_in}, {W_in}), self.dim={self.dim}")
        # assert C == self.dim, f"Channel mismatch: input C={C}, self.dim={self.dim}"


        # Convert to NHWC for internal processing
        x_nhwc = x

        # Auto padding (disabled by default for FWI)
        if self.auto_pad:
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            if pad_r > 0 or pad_b > 0:
                x_nhwc = F.pad(x_nhwc, (0, 0, 0, pad_r, 0, pad_b))
            H, W = x_nhwc.shape[1], x_nhwc.shape[2]
        else:
            assert H_in % self.n_win == 0 and W_in % self.n_win == 0, \
                f"Input size ({H_in}, {W_in}) must be divisible by n_win={self.n_win}"
            H, W = H_in, W_in

        x_patch = rearrange(x_nhwc, "b (j h) (i w) c -> b (j i) h w c",
                            j=self.n_win, i=self.n_win)

        # QKV projection
        q, kv = self.qkv(x_patch)  # q: (B, P, h, w, C_qk), kv: (B, P, h, w, C_qk+C_v)
        # Pixel-wise tokens
        q_pix = rearrange(q, 'b p2 h w c -> b p2 (h w) c')
        kv_pix = rearrange(kv, 'b p2 h w c -> (b p2) c h w')  # 标准的 PyTorch 图像格式 (NCHW)。
        kv_pix = self.kv_down(kv_pix)
        kv_pix = rearrange(kv_pix, '(b j i) c h w -> b (j i) (h w) c',
                           j=self.n_win, i=self.n_win)

        q_win = q.mean([2, 3])  # (B, P, C_qk)
        k_win = kv[..., :self.qk_dim].mean([2, 3])  # (B, P, C_qk)

        # Routing
        r_weight, r_idx = self.router(q_win, k_win)  # (B, P, topk)

        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)


        k_pix_sel = rearrange(k_pix_sel, 'b p2 k w2 (m c) -> (b p2) m c (k w2)', m=self.num_heads)
        v_pix_sel = rearrange(v_pix_sel, 'b p2 k w2 (m c) -> (b p2) m (k w2) c', m=self.num_heads)
        q_pix = rearrange(q_pix, 'b p2 w2 (m c) -> (b p2) m w2 c', m=self.num_heads)

        attn_weight = (q_pix * self.scale) @ k_pix_sel
        # softmax
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_pix_sel

        out = rearrange(out, '(b j i) m (h w) c -> b (j h) (i w) (m c)',
                        j=self.n_win, i=self.n_win,
                        h=H // self.n_win, w=W // self.n_win)

        lepe_input = rearrange(kv[..., self.qk_dim:], 'b (j i) h w c -> b c (j h) (i w)',
                               j=self.n_win, i=self.n_win).contiguous()
        lepe_out = self.lepe(lepe_input)
        lepe_out = rearrange(lepe_out, 'b c (j h) (i w) -> b (j h) (i w) c',
                             j=self.n_win, i=self.n_win)
        out = out + lepe_out

        # Output projection
        out = self.wo(out)

        # Crop padding if applied
        if self.auto_pad and (W != W_in or H != H_in):
            out = out[:, :H_in, :W_in, :].contiguous()

        # Convert back to NCHW
        # out = out.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        if ret_attn_mask:
            return out, r_weight, r_idx, attn_weight
        else:
            return out