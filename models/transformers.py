import math
import jittor as jt
import jittor.nn as nn
from typing import Optional
import os

def init_linear(l, stddev):
    nn.init.gauss_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float,
        qkv_bias: bool
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(heads=heads, n_ctx=n_ctx)
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def execute(self, x):
        x = self.c_qkv(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x


class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, heads: int, n_ctx: int):
        super().__init__()
        self.heads = heads
        self.n_ctx = n_ctx

    def execute(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = jt.split(qkv, attn_ch, dim=-1)

        weight = jt.einsum("bthc,bshc->bhts", q * scale, k * scale)
        weight = nn.softmax(weight.float(), dim=-1)
        out = jt.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)

        return out


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 1.0,
        qkv_bias: bool = True
    ):
        super().__init__()

        self.attn = MultiheadAttention(
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias
        )
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = MLP(width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width)

    def execute(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        heads: int,
        init_scale: float,
        qkv_bias: bool = True,
        n_data: Optional[int] = None,
        data_width: Optional[int] = None,
    ):
        super().__init__()
        self.n_data = n_data
        self.width = width
        self.heads = heads
        self.data_width = width if data_width is None else data_width
        self.c_q = nn.Linear(width, width, bias=qkv_bias)
        self.c_kv = nn.Linear(self.data_width, width * 2, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadCrossAttention(heads=heads, n_data=n_data)

        init_linear(self.c_q, init_scale)
        init_linear(self.c_kv, init_scale)
        init_linear(self.c_proj, init_scale)

    def execute(self, x, data):
        x = self.c_q(x)
        data = self.c_kv(data)
        x = self.attention(x, data)
        x = self.c_proj(x)
        return x


class QKVMultiheadCrossAttention(nn.Module):
    def __init__(self, *, heads: int, n_data: Optional[int] = None):

        super().__init__()
        self.heads = heads
        self.n_data = n_data

    def execute(self, q, kv):
        _, n_ctx, _ = q.shape
        bs, n_data, width = kv.shape
        attn_ch = width // self.heads // 2
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        q = q.view(bs, n_ctx, self.heads, -1)
        kv = kv.view(bs, n_data, self.heads, -1)
        k, v = jt.split(kv, attn_ch, dim=-1)

        weight = jt.einsum("bthc,bshc->bhts", q * scale, k * scale)
        weight = nn.softmax(weight.float(), dim=-1)
        out = jt.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)

        return out


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_data: Optional[int] = None,
        width: int,
        heads: int,
        data_width: Optional[int] = None,
        mlp_width_scale: int = 4,
        init_scale: float = 0.25,
        qkv_bias: bool = True,
    ):
        super().__init__()

        if data_width is None:
            data_width = width

        self.attn = MultiheadCrossAttention(
            n_data=n_data,
            width=width,
            heads=heads,
            data_width=data_width,
            init_scale=init_scale,
            qkv_bias=qkv_bias
        )
        self.ln_1 = nn.LayerNorm(width)
        self.ln_2 = nn.LayerNorm(data_width)
        self.mlp = MLP(width=width, hidden_width_scale=mlp_width_scale, init_scale=init_scale)
        self.ln_3 = nn.LayerNorm(width)

    def execute(self, x, data):
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        x = x + self.mlp(self.ln_3(x))
        return x


class MLP(nn.Module):
    def __init__(self, *,
                 width: int,
                 hidden_width_scale: int = 4,
                 init_scale: float):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * hidden_width_scale)
        self.c_proj = nn.Linear(width * hidden_width_scale, width)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def execute(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        init_scale: float = 0.25,
        qkv_bias: bool = True,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                    qkv_bias=qkv_bias,
                )
                for _ in range(layers)
            ]
        )

    def execute(self, x):
        for block in self.resblocks:
            x = block(x)
        return x
