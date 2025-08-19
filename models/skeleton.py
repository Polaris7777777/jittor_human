import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat
import numpy as np
import sys
from models.transformers import ResidualCrossAttentionBlock
from models.sal_perceiver import ShapeAsLatentPerceiverEncoder
from models.PCT.networks.cls.pct import Point_Transformer, Point_Transformer2

class SimpleSkeletonModel(nn.Module):
    
    def __init__(self, feat_dim: int, output_channels: int):
        super().__init__()
        self.feat_dim           = feat_dim
        self.output_channels    = output_channels
        
        self.transformer = Point_Transformer(output_channels=feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_channels),
        )
    
    def execute(self, vertices: jt.Var):
        vertices = vertices.permute(0, 2, 1)  # [B, 3, N]
        x = self.transformer(vertices)
        return self.mlp(x)

class PCT2SkeletonModel(nn.Module):
    
    def __init__(self, feat_dim: int, output_channels: int):
        super().__init__()
        self.feat_dim           = feat_dim
        self.output_channels    = output_channels
        
        self.transformer = Point_Transformer2(output_channels=feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_channels),
        )
    
    def execute(self, vertices: jt.Var):
        vertices = vertices.permute(0, 2, 1)
        x = self.transformer(vertices)
        return self.mlp(x)

class SALSkeletonModel(nn.Module):
    def __init__(self, 
                 output_channels:int = 66,
                 num_skeletons:int = 22,
                 num_latents=512,
                 point_feats=3,
                 embed_dim=64,
                 num_freqs=8,
                 include_pi=False,
                 width=512,
                 heads=8,
                 num_encoder_layers=8,
                 init_scale=0.25,
                 qkv_bias=True,
                 use_ln_post=False,
                 query_method=False,
                 token_num=512,
                 grad_interval=0.005,
                 use_full_input=True,
                 freeze_encoder=False
                 ):
        super().__init__()
        self.output_channels    = output_channels // num_skeletons
        self.num_skeletons      = num_skeletons
        self.width              = width
        self.encoder = ShapeAsLatentPerceiverEncoder(
            num_latents=num_latents,
            point_feats=point_feats,
            embed_dim=embed_dim,
            num_freqs=num_freqs,
            include_pi=include_pi,
            width=width,
            heads=heads,
            num_encoder_layers=num_encoder_layers,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            use_ln_post=use_ln_post,
            query_method=query_method,
            token_num=token_num,
            grad_interval=grad_interval,
            use_full_input=use_full_input,
            freeze_encoder=freeze_encoder
        )

        # self.skeleton_queries = nn.Parameter(jt.randn((num_skeletons, width)) * 0.02)
        # self.skeleton_queries.requires_grad = True
        # init.xavier_uniform_(self.skeleton_queries)

        
        self.skeleton_queries = nn.Embedding(num_skeletons, width).weight
        init.xavier_uniform_(self.skeleton_queries)

        self.cross_attn = ResidualCrossAttentionBlock(
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias
        )

        self.to_output = nn.Linear(width, self.output_channels)


    def decode(self, x, queries):
        queries = queries.unsqueeze(0).expand(x.shape[0], -1, -1)  # Expand queries to match batch size
        latents = self.cross_attn(queries, x)
        return self.to_output(latents)

    
    def execute(self, pc: jt.Var, normals: jt.Var = None):
        _, latents, _, _ = self.encoder.encode_latents(pc, feats=normals)
        x = self.decode(latents, self.skeleton_queries).view(-1, self.num_skeletons * self.output_channels)

        return x

    def load_with_skeleton_transfer(self, pretrained_path: str, num_pretrained_skeletons: int = 22):
        """
        加载模型并进行skeleton_queries的权重迁移
        
        Args:
            pretrained_path: 预训练模型路径
            num_pretrained_skeletons: 预训练模型中的骨骼数量
        """
        try:
            # 首先尝试正常加载所有兼容的权重
            pretrained_state = jt.load(pretrained_path)
            
            # 保存当前skeleton_queries的权重（防止被覆盖）
            current_skeleton_queries = self.skeleton_queries.clone()
            
            # 从预训练状态字典中移除skeleton_queries相关的权重
            skeleton_keys = [k for k in pretrained_state.keys() if 'skeleton_queries' in k]
            pretrained_skeleton_state = {}
            for key in skeleton_keys:
                pretrained_skeleton_state[key] = pretrained_state.pop(key)
            
            # 加载其他兼容的权重
            self.load_state_dict(pretrained_state)
            print("✅ Loaded compatible weights from pretrained model")
            
            # 手动处理skeleton_queries的权重迁移
            if 'skeleton_queries' in pretrained_skeleton_state:
                pretrained_queries = pretrained_skeleton_state['skeleton_queries']
                
                if pretrained_queries.shape[0] >= num_pretrained_skeletons and \
                   pretrained_queries.shape[1] == self.width:
                    
                    with jt.no_grad():
                        # 复制前num_pretrained_skeletons个权重
                        self.skeleton_queries[:num_pretrained_skeletons] = \
                            pretrained_queries[:num_pretrained_skeletons]
                        
                        # 保持剩余权重的随机初始化
                        if self.num_skeletons > num_pretrained_skeletons:
                            # 重新初始化新增的skeleton queries
                            init.xavier_uniform_(
                                self.skeleton_queries[num_pretrained_skeletons:]
                            )
                    self.skeleton_queries.requires_grad = True

                    print(f"✅ Successfully transferred skeleton_queries weights:")
                    print(f"   Transferred: {num_pretrained_skeletons} queries")
                    print(f"   Newly initialized: {self.num_skeletons - num_pretrained_skeletons} queries")
                    
                else:
                    print(f"❌ Skeleton queries shape mismatch: {pretrained_queries.shape}")
                    
        except Exception as e:
            print(f"❌ Error in model loading with skeleton transfer: {e}")
            raise e

# Factory function to create models
def create_model(model_name='pct', output_channels=66, **kwargs):
    if model_name == "pct":
        return SimpleSkeletonModel(feat_dim=256, output_channels=output_channels)
    if model_name == "pct2":
        return PCT2SkeletonModel(feat_dim=256, output_channels=output_channels)
    if model_name == "sal":
        with_normals=kwargs.get('with_normals', False)
        num_latents = kwargs.get('num_tokens', 512)
        num_tokens = kwargs.get('num_tokens', 512)
        feat_dim = kwargs.get('feat_dim', 512)
        layers = kwargs.get('encoder_layers', 8)
        point_feats = 3 if with_normals else 0
        skeletons = output_channels // 3
        return SALSkeletonModel(output_channels=output_channels,
                                num_skeletons=skeletons, 
                                point_feats=point_feats,
                                num_latents=num_latents,
                                token_num=num_tokens,
                                width=feat_dim,
                                num_encoder_layers=layers)
    raise NotImplementedError()
