import jittor as jt
from jittor import nn
import numpy as np
import math

from .codebook import VectorQuantizer, PositionalEncoding, MultiHeadAttention, FeedForward

class PoseAwareEncoder(nn.Module):
    """姿态感知的关键点编码器"""
    
    def __init__(self, num_joints=52, hidden_dim=512, embedding_dim=512, num_layers=4, num_heads=8):
        super().__init__()
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim
        
        # 输入投影层 - 包含位置和姿态信息
        self.input_projection = nn.Linear(3, hidden_dim)
        
        # 关节类型嵌入 - 区分身体、手部关节
        self.joint_type_embedding = nn.Embedding(3, hidden_dim)  # 0: body, 1: left_hand, 2: right_hand
        
        # 层次化位置编码 - 考虑骨骼层次结构
        self.hierarchical_pos_encoding = HierarchicalPositionalEncoding(hidden_dim, num_joints)
        
        # Multi-scale Transformer layers
        self.local_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 2, dropout=0.1)
            for _ in range(num_layers // 2)
        ])
        
        self.global_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout=0.1)
            for _ in range(num_layers - num_layers // 2)
        ])
        
        # 自适应池化 - 根据关节重要性加权
        self.attention_pool = AttentionPooling(hidden_dim)
        
        # 输出投影
        self.output_projection = nn.Linear(hidden_dim, embedding_dim)
        
    def execute(self, joints):
        # joints: [B, J, 3]
        batch_size, num_joints, _ = joints.shape
        
        # 投影到hidden_dim
        x = self.input_projection(joints)  # [B, J, hidden_dim]
        
        # 添加关节类型嵌入
        joint_types = self.get_joint_types(num_joints)
        type_embed = self.joint_type_embedding(joint_types).unsqueeze(0).expand(batch_size, -1, -1)
        x = x + type_embed
        
        # 层次化位置编码
        x = self.hierarchical_pos_encoding(x)
        
        # Local feature extraction
        for layer in self.local_layers:
            x = layer(x)
        
        # Global feature extraction
        for layer in self.global_layers:
            x = layer(x)
        
        # 自适应池化
        x = self.attention_pool(x)  # [B, hidden_dim]
        
        # 输出投影
        x = self.output_projection(x)  # [B, embedding_dim]
        
        return x
    
    def get_joint_types(self, num_joints):
        """获取关节类型标识"""
        if num_joints == 22:
            # 身体关节
            return jt.zeros(22, dtype=jt.int32)
        elif num_joints == 52:
            # 身体 + 手部关节 - 修复：基于format.py中的定义
            types = jt.zeros(52, dtype=jt.int32)
            # 根据format.py，前22个是身体关节，22-36是左手，37-51是右手
            types[22:37] = 1  # 左手：22-36 (15个关节)
            types[37:52] = 2  # 右手：37-51 (15个关节)
            return types
        else:
            return jt.zeros(num_joints, dtype=jt.int32)

class HierarchicalPositionalEncoding(nn.Module):
    """层次化位置编码，考虑骨骼的父子关系"""
    
    def __init__(self, d_model, num_joints):
        super().__init__()
        self.d_model = d_model
        self.num_joints = num_joints
        
        # 基础位置编码
        self.pos_encoding = PositionalEncoding(d_model, num_joints)
        
        # 层次编码 - 根据骨骼层次深度
        self.hierarchy_embedding = nn.Embedding(5, d_model)  # 最多5层深度
        
        # 骨骼链编码 - 编码关节在运动链中的位置
        self.chain_embedding = nn.Embedding(10, d_model)  # 支持不同的运动链
        
    def execute(self, x):
        # 基础位置编码
        x = self.pos_encoding(x)
        
        # 添加层次编码
        hierarchy_levels = self.get_hierarchy_levels(self.num_joints)
        hierarchy_embed = self.hierarchy_embedding(hierarchy_levels).unsqueeze(0)
        x = x + hierarchy_embed
        
        # 添加运动链编码
        chain_ids = self.get_chain_ids(self.num_joints)
        chain_embed = self.chain_embedding(chain_ids).unsqueeze(0)
        x = x + chain_embed
        
        return x
    
    def get_hierarchy_levels(self, num_joints):
        """获取关节的层次深度 - 基于format.py中的parents定义"""
        if num_joints == 22:
            # 基于format.py中的parents计算层次深度
            # parents = [None, 0, 1, 2, 3, 4, 3, 6, 7, 8, 3, 10, 11, 12, 0, 14, 15, 16, 0, 18, 19, 20]
            levels = jt.array([0, 1, 2, 3, 4, 5, 4, 5, 6, 7, 4, 5, 6, 7, 1, 2, 3, 4, 1, 2, 3, 4])
        elif num_joints == 52:
            # 身体部分的层次深度
            body_levels = jt.array([0, 1, 2, 3, 4, 5, 4, 5, 6, 7, 4, 5, 6, 7, 1, 2, 3, 4, 1, 2, 3, 4])
            
            # 手部关节的层次深度 - 基于format.py中的parents
            # 左手：22-36，父节点都是9 (l_hand)，所以层次为8
            # 手指关节：每个手指3个关节，层次递增
            left_hand_levels = jt.array([
                8, 9, 10,  # 拇指：22-24
                8, 9, 10,  # 食指：25-27  
                8, 9, 10,  # 中指：28-30
                8, 9, 10,  # 无名指：31-33
                8, 9, 10   # 小指：34-36
            ])
            
            # 右手：37-51，父节点都是13 (r_hand)，所以层次为8
            right_hand_levels = jt.array([
                8, 9, 10,  # 拇指：37-39
                8, 9, 10,  # 食指：40-42
                8, 9, 10,  # 中指：43-45
                8, 9, 10,  # 无名指：46-48
                8, 9, 10   # 小指：49-51
            ])
            
            levels = jt.concat([body_levels, left_hand_levels, right_hand_levels], dim=0)
        else:
            levels = jt.zeros(num_joints, dtype=jt.int32)
        
        # 限制最大层次为4，避免超出embedding范围
        return levels.clamp(0, 4)
    
    def get_chain_ids(self, num_joints):
        """获取关节所属的运动链ID - 基于format.py中的骨骼结构"""
        if num_joints == 22:
            # 基于解剖结构的运动链
            # 0: spine, 1: left_arm, 2: right_arm, 3: left_leg, 4: right_leg
            chain_ids = jt.array([
                0, 0, 0, 0, 0, 0,     # 脊柱链：0-5 (hips到head)
                1, 1, 1, 1,           # 左臂链：6-9 (l_shoulder到l_hand)  
                2, 2, 2, 2,           # 右臂链：10-13 (r_shoulder到r_hand)
                3, 3, 3, 3,           # 左腿链：14-17 (l_upper_leg到l_toe_base)
                4, 4, 4, 4            # 右腿链：18-21 (r_upper_leg到r_toe_base)
            ])
        elif num_joints == 52:
            # 身体部分的运动链
            body_chains = jt.array([
                0, 0, 0, 0, 0, 0,     # 脊柱链：0-5
                1, 1, 1, 1,           # 左臂链：6-9
                2, 2, 2, 2,           # 右臂链：10-13
                3, 3, 3, 3,           # 左腿链：14-17
                4, 4, 4, 4            # 右腿链：18-21
            ])
            
            # 左手手指运动链：5-9 (每个手指一个链)
            left_hand_chains = jt.array([
                5, 5, 5,  # 左拇指：22-24
                6, 6, 6,  # 左食指：25-27
                7, 7, 7,  # 左中指：28-30
                8, 8, 8,  # 左无名指：31-33
                9, 9, 9   # 左小指：34-36
            ])
            
            # 右手手指运动链：重复使用5-9
            right_hand_chains = jt.array([
                5, 5, 5,  # 右拇指：37-39
                6, 6, 6,  # 右食指：40-42
                7, 7, 7,  # 右中指：43-45
                8, 8, 8,  # 右无名指：46-48
                9, 9, 9   # 右小指：49-51
            ])
            
            chain_ids = jt.concat([body_chains, left_hand_chains, right_hand_chains], dim=0)
        else:
            chain_ids = jt.zeros(num_joints, dtype=jt.int32)
        
        return chain_ids.clamp(0, 9)

class AttentionPooling(nn.Module):
    """注意力池化，根据关节重要性进行加权"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.1)
        
    def execute(self, x):
        # x: [B, J, hidden_dim]
        attn_weights = self.attention(x)  # [B, J, 1]
        attn_weights = nn.softmax(attn_weights, dim=1)  # [B, J, 1]
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        pooled = jt.sum(x * attn_weights, dim=1)  # [B, hidden_dim]
        
        return pooled

class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def execute(self, x, mask=None):
        # 自注意力 + 残差连接
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

def create_pose_aware_keypoint_vqvae(num_joints=52,
                                    embedding_dim=512,
                                    num_embeddings=2048,
                                    hidden_dim=512,
                                    num_layers=4,
                                    num_heads=8,
                                    commitment_cost=0.25):
    """创建姿态感知的关键点VQ-VAE模型"""
    from .codebook import KeypointVQVAE, KeypointDecoder, VectorQuantizer
    
    # 使用姿态感知编码器
    encoder = PoseAwareEncoder(
        num_joints=num_joints,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads
    )
    
    # VQ层
    vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
    
    # 解码器保持不变
    decoder = KeypointDecoder(
        num_joints=num_joints,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_heads=num_heads
    )
    
    # 创建模型
    model = KeypointVQVAE.__new__(KeypointVQVAE)
    model.__init__(num_joints, embedding_dim, num_embeddings, hidden_dim, num_layers, num_heads, commitment_cost)
    model.encoder = encoder
    
    return model
