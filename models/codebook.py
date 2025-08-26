import jittor as jt
from jittor import nn
import numpy as np
import math

class VectorQuantizer(nn.Module):
    """Vector Quantization layer for codebook learning"""
    
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        
        # Initialize embedding vectors with much better initialization
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # 使用Xavier初始化，确保更好的分布
        limit = np.sqrt(6.0 / (num_embeddings + embedding_dim))
        init_weight = np.random.uniform(-limit, limit, (num_embeddings, embedding_dim)).astype(np.float32)
        self.embedding.weight.data = init_weight
        
        # 用于EMA更新的缓冲区 - 作为普通属性，不参与梯度计算
        self.cluster_size = np.zeros(num_embeddings, dtype=np.float32)
        self.embed_avg = init_weight.copy()
        
    def execute(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # 添加噪声以防止完全相同的输入
        if self.training:
            noise = jt.randn_like(flat_input) * 0.01
            flat_input = flat_input + noise
        
        # Fix: Manual distance calculation instead of jt.cdist
        # Calculate distances using L2 norm: ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
        distances = (jt.sum(flat_input**2, dim=1, keepdims=True) 
                    + jt.sum(self.embedding.weight**2, dim=1)
                    - 2 * jt.matmul(flat_input, self.embedding.weight.t()))
        
        # Find closest embedding
        encoding_indices = jt.argmin(distances, dim=1)
        if isinstance(encoding_indices, tuple):
            encoding_indices = encoding_indices[0]
            
        # One-hot encoding
        encodings = jt.zeros(encoding_indices.shape[0], self.num_embeddings)
        ones_tensor = jt.ones_like(encoding_indices.unsqueeze(1)).float()
        encodings.scatter_(1, encoding_indices.unsqueeze(1), ones_tensor)
        
        # Quantize
        quantized = jt.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # 强化版损失函数
        e_latent_loss = jt.mean((quantized.detach() - inputs)**2)
        q_latent_loss = jt.mean((quantized - inputs.detach())**2)
        
        # 大幅增强多样性损失
        diversity_loss = self._compute_diversity_loss(encodings)
        entropy_loss = self._compute_entropy_loss(encodings)
        usage_loss = self._compute_usage_loss(encodings)
        
        # 组合损失，大幅提高多样性权重
        loss = (q_latent_loss + 
                self.commitment_cost * e_latent_loss + 
                0.5 * diversity_loss + 
                0.3 * entropy_loss + 
                0.2 * usage_loss)
        
        # Straight through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Calculate perplexity
        avg_probs = jt.mean(encodings, dim=0)
        perplexity = jt.exp(-jt.sum(avg_probs * jt.log(avg_probs + 1e-10)))
        
        # EMA更新codebook (在训练时)
        if self.training:
            self._ema_update(encodings, flat_input)
        
        return quantized, loss, encoding_indices, perplexity
    
    def _compute_diversity_loss(self, encodings):
        """计算多样性损失，强烈鼓励使用更多的codebook entries"""
        usage_freq = jt.mean(encodings, dim=0)
        
        # 计算使用的codebook数量
        used_codes = jt.sum(usage_freq > 1e-8).float()
        max_codes = float(self.num_embeddings)
        
        # 鼓励使用更多codes
        diversity_ratio = used_codes / max_codes
        diversity_loss = 1.0 - diversity_ratio
        
        return diversity_loss
    
    def _compute_entropy_loss(self, encodings):
        """计算熵损失，鼓励均匀分布"""
        usage_freq = jt.mean(encodings, dim=0)
        
        # 避免log(0)
        usage_freq = usage_freq + 1e-10
        
        # 计算熵
        entropy = -jt.sum(usage_freq * jt.log(usage_freq))
        max_entropy = jt.log(jt.array(float(self.num_embeddings)))
        
        # 归一化熵
        normalized_entropy = entropy / max_entropy
        
        # 返回负熵作为损失（最大化熵）
        return 1.0 - normalized_entropy
    
    def _compute_usage_loss(self, encodings):
        """计算使用损失，惩罚过度集中使用某些codes"""
        usage_freq = jt.mean(encodings, dim=0)
        
        # 计算使用频率的方差，鼓励均匀使用
        mean_usage = 1.0 / self.num_embeddings
        variance = jt.mean((usage_freq - mean_usage) ** 2)
        
        return variance * self.num_embeddings
    
    def _ema_update(self, encodings, flat_input):
        """EMA更新codebook"""
        # 转换为numpy进行EMA更新，避免梯度计算
        encodings_np = encodings.detach().numpy()
        flat_input_np = flat_input.detach().numpy()
        
        # 计算每个code的使用次数
        cluster_size = np.sum(encodings_np, axis=0)
        
        # EMA更新cluster size
        self.cluster_size = self.cluster_size * self.decay + cluster_size * (1 - self.decay)
        
        # 计算每个cluster的平均值
        embed_sum = np.dot(encodings_np.T, flat_input_np)
        self.embed_avg = self.embed_avg * self.decay + embed_sum * (1 - self.decay)
        
        # 更新embedding weights
        cluster_size_stable = self.cluster_size + 1e-5
        embed_normalized = self.embed_avg / cluster_size_stable[:, np.newaxis]
        
        # 平滑更新权重
        current_weight = self.embedding.weight.data
        if isinstance(current_weight, jt.Var):
            current_weight = current_weight.numpy()
        
        new_weight = current_weight * 0.9 + embed_normalized * 0.1
        self.embedding.weight.data = new_weight

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def execute(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # 线性变换
        Q = self.w_q(query)  # [B, seq_len, d_model]
        K = self.w_k(key)
        V = self.w_v(value)
        
        # 重塑为多头
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # [B, num_heads, seq_len, d_k]
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        scores = jt.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Fix: use nn.softmax instead of jt.softmax
        attn_weights = nn.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = jt.matmul(attn_weights, V)  # [B, num_heads, seq_len, d_k]
        
        # 拼接多头 - Fix: remove .contiguous() as it's not available in Jittor
        context = context.transpose(1, 2).view(batch_size, -1, self.d_model)
        
        return self.w_o(context)

class FeedForward(nn.Module):
    """前馈网络"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def execute(self, x):
        # Fix: use nn.relu instead of jt.relu
        return self.linear2(self.dropout(nn.relu(self.linear1(x))))

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

class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def execute(self, x, memory, tgt_mask=None, memory_mask=None):
        # 自注意力
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 交叉注意力
        attn_output = self.cross_attn(x, memory, memory, memory_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = jt.zeros(max_len, d_model)
        position = jt.arange(0, max_len, dtype=jt.float32).unsqueeze(1)
        div_term = jt.exp(jt.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = jt.sin(position * div_term)
        pe[:, 1::2] = jt.cos(position * div_term)
        
        # 使用下划线前缀避免被当作参数
        self._pe = pe
    
    def execute(self, x):
        # x: [B, seq_len, d_model]
        seq_len = x.shape[1]
        pe_slice = self._pe[:seq_len, :].unsqueeze(0)
        if hasattr(x, 'device'):
            pe_slice = pe_slice.to(x.device)
        return x + pe_slice

class KeypointEncoder(nn.Module):
    """Transformer编码器for关键点位置"""
    
    def __init__(self, num_joints=22, hidden_dim=512, embedding_dim=512, num_layers=4, num_heads=8):
        super().__init__()
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim
        
        # 输入投影层
        self.input_projection = nn.Linear(3, hidden_dim)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=num_joints)
        
        # Transformer编码器层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        # 输出投影层
        self.output_projection = nn.Linear(hidden_dim, embedding_dim)
        
    def execute(self, joints):
        # joints: [B, J, 3]
        batch_size, num_joints, _ = joints.shape
        
        # 投影到hidden_dim
        x = self.input_projection(joints)  # [B, J, hidden_dim]
        
        # 添加位置编码
        x = self.pos_encoding(x)
        
        # Transformer编码
        for layer in self.layers:
            x = layer(x)
        
        # 全局平均池化：手动实现
        x = jt.mean(x, dim=1)  # [B, hidden_dim]
        
        # 输出投影
        x = self.output_projection(x)  # [B, embedding_dim]
        
        return x

class KeypointDecoder(nn.Module):
    """Transformer解码器for关键点位置"""
    
    def __init__(self, num_joints=22, hidden_dim=512, embedding_dim=512, num_layers=4, num_heads=8):
        super().__init__()
        self.num_joints = num_joints
        self.hidden_dim = hidden_dim
        
        # 输入投影层
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        
        # 可学习的关节查询向量 - 保持为参数
        self.joint_queries = jt.randn(num_joints, hidden_dim)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=num_joints)
        
        # Transformer解码器层
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        # 输出投影层
        self.output_projection = nn.Linear(hidden_dim, 3)
        
    def execute(self, quantized):
        # quantized: [B, embedding_dim]
        batch_size = quantized.shape[0]
        
        # 输入投影为memory
        memory = self.input_projection(quantized)  # [B, hidden_dim]
        memory = memory.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # 扩展关节查询向量到batch
        tgt = self.joint_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [B, J, hidden_dim]
        
        # 添加位置编码
        tgt = self.pos_encoding(tgt)
        
        # Transformer解码
        for layer in self.layers:
            tgt = layer(tgt, memory)
        
        # 输出投影到3D坐标
        joints = self.output_projection(tgt)  # [B, J, 3]
        
        return joints

class KeypointVQVAE(nn.Module):
    """VQ-VAE for keypoint positions with Transformer encoder/decoder"""
    
    def __init__(self, 
                 num_joints=22,
                 embedding_dim=512,
                 num_embeddings=1024,
                 hidden_dim=512,
                 num_layers=4,
                 num_heads=8,
                 commitment_cost=0.25):
        super().__init__()
        
        self.encoder = KeypointEncoder(
            num_joints=num_joints, 
            hidden_dim=hidden_dim, 
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads
        )
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = KeypointDecoder(
            num_joints=num_joints, 
            hidden_dim=hidden_dim, 
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads
        )
        
    def execute(self, joints, return_indices=False):
        # Encode
        encoded = self.encoder(joints)
        
        # Quantize
        quantized, vq_loss, indices, perplexity = self.vq_layer(encoded)
        
        # Decode
        reconstructed = self.decoder(quantized)
        
        if return_indices:
            return reconstructed, vq_loss, indices, perplexity
        else:
            return reconstructed, vq_loss

def create_keypoint_vqvae(num_joints=22,
                         embedding_dim=512,
                         num_embeddings=1024,
                         hidden_dims=None,  # 保持兼容性，但不使用
                         hidden_dim=512,
                         num_layers=4,
                         num_heads=8,
                         commitment_cost=0.25):
    """Create a keypoint VQ-VAE model with Transformer"""
    return KeypointVQVAE(
        num_joints=num_joints,
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        commitment_cost=commitment_cost
    )
    
