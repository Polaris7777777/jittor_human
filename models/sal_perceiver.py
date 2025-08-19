import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat
import numpy as np
import sys
import math
from typing import Optional, Tuple
import random
from jittor import einops 

from models.transformers import Transformer, MLP, ResidualCrossAttentionBlock
from models.pointnet_ops import FurthestPointSampler as fps

class FourierEmbedder(nn.Module):
    """The sin/cosine positional embedding. Given an input tensor `x` of shape [n_batch, ..., c_dim], it converts
    each feature dimension of `x[..., i]` into:
        [
            sin(x[..., i]),
            sin(f_1*x[..., i]),
            sin(f_2*x[..., i]),
            ...
            sin(f_N * x[..., i]),
            cos(x[..., i]),
            cos(f_1*x[..., i]),
            cos(f_2*x[..., i]),
            ...
            cos(f_N * x[..., i]),
            x[..., i]     # only present if include_input is True.
        ], here f_i is the frequency.

    Denote the space is [0 / num_freqs, 1 / num_freqs, 2 / num_freqs, 3 / num_freqs, ..., (num_freqs - 1) / num_freqs].
    If logspace is True, then the frequency f_i is [2^(0 / num_freqs), ..., 2^(i / num_freqs), ...];
    Otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1)].

    Args:
        num_freqs (int): the number of frequencies, default is 6;
        logspace (bool): If logspace is True, then the frequency f_i is [..., 2^(i / num_freqs), ...],
            otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1)];
        input_dim (int): the input dimension, default is 3;
        include_input (bool): include the input tensor or not, default is True.

    Attributes:
        frequencies (torch.Tensor): If logspace is True, then the frequency f_i is [..., 2^(i / num_freqs), ...],
                otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1);

        out_dim (int): the embedding size, if include_input is True, it is input_dim * (num_freqs * 2 + 1),
            otherwise, it is input_dim * num_freqs * 2.

    """

    def __init__(self,
                 num_freqs: int = 6,
                 logspace: bool = True,
                 input_dim: int = 3,
                 include_input: bool = True,
                 include_pi: bool = True) -> None:

        """The initialization"""

        super().__init__()

        if logspace:
            frequencies = 2.0 ** jt.arange(num_freqs)
        else:
            frequencies = jt.linspace(1.0, 2.0 ** (num_freqs - 1), num_freqs)

        if include_pi:
            frequencies *= math.pi

        self.frequencies = frequencies.stop_grad()
        # self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs

        self.out_dim = self.get_dims(input_dim)

    def get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        out_dim = input_dim * (self.num_freqs * 2 + temp)

        return out_dim

    def execute(self, x: jt.Var) -> jt.Var:
        """ Forward process.

        Args:
            x: tensor of shape [..., dim]

        Returns:
            embedding: an embedding of `x` of shape [..., dim * (num_freqs * 2 + temp)]
                where temp is 1 if include_input is True and 0 otherwise.
        """

        if self.num_freqs > 0:
            embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)
            if self.include_input:
                return concat([x, embed.sin(), embed.cos()], dim=-1)
            else:
                return concat([embed.sin(), embed.cos()], dim=-1)
        else:
            return x


class CrossAttentionEncoder(nn.Module):

    def __init__(self, *,
                 num_latents: int,
                 fourier_embedder: FourierEmbedder,
                 point_feats: int,
                 width: int,
                 heads: int,
                 layers: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 use_ln_post: bool = False,
                 use_checkpoint: bool = False,
                 query_method: bool = False,
                 use_full_input: bool = True,
                 token_num: int = 256,
                 no_query: bool=False):

        super().__init__()

        self.query_method = query_method
        self.token_num = token_num
        self.use_full_input = use_full_input

        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents

        if no_query:
            self.query = None
        else:
            self.query = nn.Parameter(jt.randn((num_latents, width)) * 0.02)

        self.fourier_embedder = fourier_embedder
        self.input_proj = nn.Linear(self.fourier_embedder.out_dim + point_feats, width)
        self.cross_attn = ResidualCrossAttentionBlock(
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias
        )

        self.self_attn = Transformer(
            n_ctx=num_latents,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias
        )

        self.fps = fps(token_num)

        if use_ln_post:
            self.ln_post = nn.LayerNorm(width)
        else:
            self.ln_post = None

    def execute(self, pc, feats=None):
        if self.query_method:
            token_num = self.num_latents
            bs = pc.shape[0] 
            data = self.fourier_embedder(pc) 
            if feats is not None: 
                data = concat([data, feats], dim=-1)
            data = self.input_proj(data) 

            query = einops.repeat(self.query, "m c -> b m c", b=bs) 

            latents = self.cross_attn(query, data)
            latents = self.self_attn(latents)

            if self.ln_post is not None:
                latents = self.ln_post(latents)

            pre_pc = None
        else:
            if isinstance(self.token_num, int):
                token_num = self.token_num
            else:
                token_num = random.choice(self.token_num)

            if self.training:
                rng = np.random.default_rng()
            else:
                rng = np.random.default_rng(seed=123)
            ind = rng.choice(pc.shape[1], token_num * 4, replace=token_num * 4 > pc.shape[1])

            pre_pc = pc[:,ind,:]
            pre_feats = feats[:,ind,:] if feats is not None else None

            B, N, D = pre_pc.shape           
            C = pre_feats.shape[-1] if feats is not None else 0
            ###### fps
            # pos = pre_pc.view(B*N, D)
            # pos_feats = pre_feats.view(B*N, C)
            # batch = jt.arange(B)
            # batch = jt.repeat_interleave(batch, N)
            # idx = self.fps(pos, batch, ratio=1. / 4, random_start=self.training)

            # sampled_pc = pos[idx]
            # sampled_pc = sampled_pc.view(B, -1, 3)

            # sampled_feats = pos_feats[idx]
            # sampled_feats = sampled_feats.view(B, -1, C)

            # import pdb; pdb.set_trace()
            sampled_pc, inds = self.fps(pre_pc, return_index=True)

            batch_inds = jt.arange(B).unsqueeze(1)  # [B, 1]
            sampled_feats = pre_feats[batch_inds, inds] if feats is not None else None

            ######
            if self.use_full_input:
                data = self.fourier_embedder(pc) 
            else:
                data = self.fourier_embedder(pre_pc) 
            
            if feats is not None: 
                if not self.use_full_input:
                    feats = pre_feats
                data = concat([data, feats], dim=-1) 
            data = self.input_proj(data) 

            sampled_data = self.fourier_embedder(sampled_pc)
            if feats is not None: 
                sampled_data = concat([sampled_data, sampled_feats], dim=-1) 
            sampled_data = self.input_proj(sampled_data) 

            latents = self.cross_attn(sampled_data, data) 
            latents = self.self_attn(latents)

            if self.ln_post is not None:
                latents = self.ln_post(latents)
            
            if feats is not None:
                pre_pc = concat([pre_pc, pre_feats], dim=-1)

        return latents, pc, token_num, pre_pc



#####################################################
# a simplified verstion of perceiver encoder
#####################################################

class ShapeAsLatentModule(nn.Module):
    latent_shape: Tuple[int, int]

    def __init__(self, *args, **kwargs):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        raise NotImplementedError

    def query_geometry(self, *args, **kwargs):
        raise NotImplementedError

class ShapeAsLatentPerceiverEncoder(ShapeAsLatentModule):
    def __init__(self, *,
                 num_latents: int,
                 point_feats: int = 3,
                 embed_dim: int = 0,
                 num_freqs: int = 8,
                 include_pi: bool = True,
                 width: int,
                 heads: int,
                 num_encoder_layers: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 use_ln_post: bool = False,
                 query_method: bool = False,
                 token_num: int = 256,
                 grad_interval: float = 0.005,
                 use_full_input: bool = True,
                 freeze_encoder: bool = False
                 ):

        super().__init__()

        self.num_latents = num_latents
        self.grad_interval = grad_interval
        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)

        init_scale = init_scale * math.sqrt(1.0 / width)
        self.encoder = CrossAttentionEncoder(
            fourier_embedder=self.fourier_embedder,
            num_latents=num_latents,
            point_feats=point_feats,
            width=width,
            heads=heads,
            layers=num_encoder_layers,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_ln_post=use_ln_post,
            query_method=query_method,
            use_full_input=use_full_input,
            token_num=token_num,
            no_query=True,
        )

        self.embed_dim = embed_dim
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("freeze encoder")
        self.width = width

    def encode_latents(self, pc: jt.Var, feats: jt.Var = None):

        x, _, token_num, pre_pc = self.encoder(pc, feats)

        shape_embed = x[:, 0]
        latents = x

        return shape_embed, latents, token_num, pre_pc

    def execute(self):
        raise NotImplementedError()