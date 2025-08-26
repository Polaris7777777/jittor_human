import jittor as jt
import numpy as np
from jittor import nn
import jittor.nn as F
import matplotlib.pyplot as plt

class VectorQuantizer(nn.Module):
    """
    Vector Quantization module for VQVAE
    
    Args:
        num_embeddings (int): Size of the codebook
        embedding_dim (int): Dimension of each codebook vector
        commitment_cost (float): Weight for commitment loss
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        jt.init.uniform_(self.embeddings.weight, -1.0 / self.num_embeddings, 1.0 / self.num_embeddings)
        
        # For EMA updates
        self.register_buffer('ema_cluster_size', jt.zeros(num_embeddings))
        self.register_buffer('ema_w', jt.clone(self.embeddings.weight.data))
        self.register_buffer('cluster_size', jt.zeros(num_embeddings))
        
        self.decay = 0.99
        self.epsilon = 1e-5
        
    def execute(self, inputs):
        # Flatten input
        input_shape = inputs.shape
        flat_input = inputs.reshape(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (jt.sum(flat_input**2, dim=1, keepdims=True) 
                   + jt.sum(self.embeddings.weight**2, dim=1)
                   - 2 * jt.matmul(flat_input, self.embeddings.weight.transpose(0, 1)))
        
        # Encoding - fix jt.argmin usage
        encoding_indices = jt.argmin(distances, dim=1)[0]  # Get the indices, not the tuple
        encodings = jt.zeros(encoding_indices.shape[0], self.num_embeddings)
        # Fix scatter_ call - use tensor instead of scalar, fix ones_like call
        ones_tensor = jt.ones_like(encoding_indices).float().unsqueeze(1)
        encodings.scatter_(1, encoding_indices.reshape(-1, 1), ones_tensor)
        
        # Quantize
        quantized = jt.matmul(encodings, self.embeddings.weight)
        quantized = quantized.reshape(input_shape)
        
        # Compute losses
        e_latent_loss = jt.mean((quantized.detach() - inputs) ** 2)
        q_latent_loss = jt.mean((quantized - inputs.detach()) ** 2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # EMA update (training only)
        if self.training:
            self._ema_update(flat_input, encodings)
        
        # Straight through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Return quantized vectors and loss
        avg_probs = jt.mean(encodings, dim=0)
        perplexity = jt.exp(-jt.sum(avg_probs * jt.log(avg_probs + 1e-10)))
        
        return quantized, loss, encoding_indices, perplexity
    
    def _ema_update(self, flat_input, encodings):
        """Update codebook using Exponential Moving Average"""
        self.cluster_size = self.cluster_size * self.decay + (1 - self.decay) * jt.sum(encodings, dim=0)
        
        # Laplace smoothing
        n = jt.sum(self.cluster_size)
        self.cluster_size = ((self.cluster_size + self.epsilon) / 
                             (n + self.num_embeddings * self.epsilon) * n)
        
        dw = jt.matmul(encodings.transpose(0, 1), flat_input)
        self.ema_w = self.ema_w * self.decay + (1 - self.decay) * dw
        
        # Update embeddings
        self.embeddings.weight = self.ema_w / self.cluster_size.unsqueeze(1)


class PointEncoder(nn.Module):
    """
    Point cloud encoder based on PointNet-like architecture
    """
    def __init__(self, input_dim=3, embedding_dim=512, hidden_dims=[128, 256, 512]):
        super(PointEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Point-wise MLP layers
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        
        self.point_mlp = nn.Sequential(*layers)
        
        # Final projection to embedding dim
        self.projection = nn.Linear(prev_dim, embedding_dim)
        
    def execute(self, x):
        """
        x: Point cloud [B, N, 3]
        """
        B, N, C = x.shape
        
        # Reshape for point-wise processing: [B*N, C]
        x_flat = x.reshape(-1, C)
        
        # Process points through MLP
        features = self.point_mlp(x_flat)  # [B*N, hidden_dim[-1]]
        
        # Reshape back: [B, N, hidden_dim[-1]]
        features = features.reshape(B, N, -1)
        
        # Max pooling over points for permutation invariance
        features = jt.max(features, dim=1)[0]  # [B, hidden_dim[-1]]
        
        # Project to embedding dimension
        embedding = self.projection(features)  # [B, embedding_dim]
        
        return embedding


class ResidualBlock(nn.Module):
    """
    Residual block for the encoder/decoder
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Linear(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
    def execute(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = F.relu(out)
        
        return out


class MLP(nn.Module):
    """
    Simple MLP for decoding embeddings to joint positions
    """
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
        
    def execute(self, x):
        return self.model(x)


class VQVAE(nn.Module):
    """
    VQVAE model for predicting skeleton joints from point clouds.
    
    Args:
        point_dim (int): Dimension of input points (typically 3)
        embedding_dim (int): Dimension of the latent embedding
        num_embeddings (int): Size of the codebook
        hidden_dims (list): List of hidden dimensions for encoder and decoder
        num_joints (int): Number of joints to predict
        commitment_cost (float): Weight for commitment loss
    """
    def __init__(self, 
                 point_dim=3,
                 embedding_dim=512, 
                 num_embeddings=1024,
                 hidden_dims=[128, 256, 512],
                 num_joints=52,
                 commitment_cost=0.25):
        super(VQVAE, self).__init__()
        
        self.point_dim = point_dim
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.hidden_dims = hidden_dims
        self.num_joints = num_joints
        
        # Point cloud encoder
        self.encoder = PointEncoder(
            input_dim=point_dim,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims
        )
        
        # Vector quantizer
        self.vq = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost
        )
        
        # Skeleton joint decoder
        self.decoder = MLP(
            input_dim=embedding_dim,
            hidden_dims=hidden_dims[::-1],  # Reverse hidden dims for decoder
            output_dim=num_joints * 3  # Each joint has 3 coordinates
        )
        
    def execute(self, points, return_indices=False):
        """
        Forward pass through the VQVAE
        
        Args:
            points (Tensor): Point cloud of shape [B, N, 3]
            return_indices (bool): Whether to return codebook indices
            
        Returns:
            joints (Tensor): Predicted joint positions of shape [B, J, 3]
            vq_loss (Tensor): Vector quantization loss
            indices (Tensor, optional): Codebook indices
            perplexity (Tensor, optional): Codebook usage metric
        """
        # Encode point cloud
        z = self.encoder(points)
        
        # Vector quantization
        quantized, vq_loss, indices, perplexity = self.vq(z)
        
        # Decode to skeleton joints
        joints = self.decoder(quantized)
        joints = joints.reshape(-1, self.num_joints, 3)
        
        if return_indices:
            return joints, vq_loss, indices, perplexity
        else:
            return joints, vq_loss
    
    def encode(self, points):
        """Encode point cloud to codebook indices"""
        z = self.encoder(points)
        _, _, indices, _ = self.vq(z)
        return indices
    
    def decode(self, indices):
        """Decode codebook indices to joints"""
        # Convert indices to one-hot
        encodings = jt.zeros(indices.shape[0], self.num_embeddings)
        # Fix scatter_ call here too
        ones_tensor = jt.ones_like(indices).float().unsqueeze(1)
        encodings.scatter_(1, indices.reshape(-1, 1), ones_tensor)
        
        # Get embedding from codebook
        quantized = jt.matmul(encodings, self.vq.embeddings.weight)
        
        # Decode to joints
        joints = self.decoder(quantized)
        return joints.reshape(-1, self.num_joints, 3)
        
    def save(self, path):
        """Save model weights"""
        self.save_parameters(path)
        
    def load(self, path):
        """Load model weights"""
        self.load_parameters(path)


class EnhancedVQVAE(nn.Module):
    """
    Enhanced VQVAE with additional features for the B competition:
    - Better handling of arbitrary poses
    - Support for 52 joints (including hand joints)
    - Stronger point encoder with cross-attention
    
    Args:
        point_dim (int): Dimension of input points (typically 3)
        embedding_dim (int): Dimension of the latent embedding
        num_embeddings (int): Size of the codebook
        hidden_dims (list): List of hidden dimensions for encoder and decoder
        num_joints (int): Number of joints to predict
        commitment_cost (float): Weight for commitment loss
        use_transformer (bool): Whether to use transformer in encoder
        transformer_layers (int): Number of transformer layers
        transformer_heads (int): Number of attention heads
    """
    def __init__(self, 
                 point_dim=3,
                 embedding_dim=512, 
                 num_embeddings=1024,
                 hidden_dims=[128, 256, 512],
                 num_joints=52,
                 commitment_cost=0.25,
                 use_transformer=True,
                 transformer_layers=4,
                 transformer_heads=8):
        super(EnhancedVQVAE, self).__init__()
        
        self.point_dim = point_dim
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.hidden_dims = hidden_dims
        self.num_joints = num_joints
        self.use_transformer = use_transformer
        
        # Point feature extraction - support both 3D points and 6D (points+normals)
        self.point_feature = nn.Sequential(
            nn.Conv1d(point_dim * 2, 64, 1),  # point_dim * 2 to support normals
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, hidden_dims[-2], 1),
            nn.BatchNorm1d(hidden_dims[-2]),
            nn.ReLU()
        )
        
        # Alternative point feature for 3D-only input
        self.point_feature_3d = nn.Sequential(
            nn.Conv1d(point_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, hidden_dims[-2], 1),
            nn.BatchNorm1d(hidden_dims[-2]),
            nn.ReLU()
        )
        
        if use_transformer:
            # Self-attention layers for better point cloud understanding
            self.transformer_layers = nn.ModuleList([
                TransformerLayer(hidden_dims[-2], transformer_heads)
                for _ in range(transformer_layers)
            ])
        
        # Final encoder layers
        self.encoder_final = nn.Sequential(
            nn.Linear(hidden_dims[-2], hidden_dims[-1]),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], embedding_dim)
        )
        
        # Vector quantizer
        self.vq = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost
        )
        
        # Hierarchical skeleton decoder for better bone structure
        self.decoder_trunk = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[-1]),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], hidden_dims[-2]),
            nn.BatchNorm1d(hidden_dims[-2]),
            nn.ReLU()
        )
        
        # Body decoder (root joints) - 根据实际需要的关节数量调整
        body_joint_count = min(22, num_joints)
        self.decoder_body = nn.Sequential(
            nn.Linear(hidden_dims[-2], hidden_dims[-3] if len(hidden_dims) > 2 else 128),
            nn.ReLU(),
            nn.Linear(hidden_dims[-3] if len(hidden_dims) > 2 else 128, body_joint_count * 3)
        )
        
        # Hand decoder (if needed)
        if num_joints > 22:
            hand_joint_count = num_joints - 22
            self.decoder_hands = nn.Sequential(
                nn.Linear(hidden_dims[-2], hidden_dims[-3] if len(hidden_dims) > 2 else 128),
                nn.ReLU(),
                nn.Linear(hidden_dims[-3] if len(hidden_dims) > 2 else 128, hand_joint_count * 3)
            )
        
    def execute(self, points, normals=None, return_indices=False):
        """
        Forward pass through the Enhanced VQVAE
        
        Args:
            points (Tensor): Point cloud of shape [B, N, 3]
            normals (Tensor, optional): Point normals of shape [B, N, 3]
            return_indices (bool): Whether to return codebook indices
            
        Returns:
            joints (Tensor): Predicted joint positions of shape [B, J, 3]
            vq_loss (Tensor): Vector quantization loss
            indices (Tensor, optional): Codebook indices
            perplexity (Tensor, optional): Codebook usage metric
        """
        
        B, N, _ = points.shape
        # import pdb; pdb.set_trace()
        # Extract point features
        x = points.transpose(1, 2)  # [B, 3, N]
        if normals is not None:
            n = normals.transpose(1, 2)  # [B, 3, N]
            x = jt.concat([x, n], dim=1)  # [B, 6, N]
            point_feat = self.point_feature(x)  # Use 6D feature extractor
        else:
            point_feat = self.point_feature_3d(x)  # Use 3D feature extractor
        
        if self.use_transformer:
            # Apply transformer layers for better feature extraction
            point_feat = point_feat.transpose(1, 2)  # [B, N, hidden_dims[-2]]
            for layer in self.transformer_layers:
                point_feat = layer(point_feat)
            
            # Global feature through max pooling
            global_feat = jt.max(point_feat, dim=1) # [B, hidden_dims[-2]]
        else:
            # Simple max pooling for global feature
            global_feat = jt.max(point_feat, dim=2)  # [B, hidden_dims[-2]]
        
        # Final encoding
        z = self.encoder_final(global_feat)  # [B, embedding_dim]
        
        # Vector quantization
        quantized, vq_loss, indices, perplexity = self.vq(z)
        # Decode trunk features
        trunk_features = self.decoder_trunk(quantized)  # [B, hidden_dims[-2]]
        
        # Decode body joints
        body_joints = self.decoder_body(trunk_features)  # [B, body_joint_count * 3]
        body_joint_count = min(22, self.num_joints)
        body_joints = body_joints.reshape(B, body_joint_count, 3)
        
        # Decode hand joints if needed
        if self.num_joints > 22:
            hand_joints = self.decoder_hands(trunk_features)  # [B, hand_joint_count * 3]
            hand_joint_count = self.num_joints - 22
            hand_joints = hand_joints.reshape(B, hand_joint_count, 3)
            joints = jt.concat([body_joints, hand_joints], dim=1)
        else:
            joints = body_joints
            
        if return_indices:
            return joints, vq_loss, indices, perplexity
        else:
            return joints, vq_loss
    
    def decode(self, indices):
        """Decode codebook indices to joints"""
        # Convert indices to one-hot
        B = indices.shape[0]
        encodings = jt.zeros(indices.shape[0], self.num_embeddings)
        # Fix scatter_ call here too
        ones_tensor = jt.ones_like(indices).float().unsqueeze(1)
        encodings.scatter_(1, indices.reshape(-1, 1), ones_tensor)
        
        # Get embedding from codebook
        quantized = jt.matmul(encodings, self.vq.embeddings.weight)
        
        # Decode trunk features
        trunk_features = self.decoder_trunk(quantized)
        
        # Decode body joints
        body_joints = self.decoder_body(trunk_features)
        body_joint_count = min(22, self.num_joints)
        body_joints = body_joints.reshape(B, body_joint_count, 3)
        
        # Decode hand joints if needed
        if self.num_joints > 22:
            hand_joints = self.decoder_hands(trunk_features)
            hand_joint_count = self.num_joints - 22
            hand_joints = hand_joints.reshape(B, hand_joint_count, 3)
            joints = jt.concat([body_joints, hand_joints], dim=1)
        else:
            joints = body_joints
            
        return joints


class TransformerLayer(nn.Module):
    """Simple Transformer layer with self-attention"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.attention = MultiHeadAttention(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def execute(self, x):
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention module"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)
        
    def execute(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear projections
        q = self.q_proj(query).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scores = jt.matmul(q, k.transpose(2, 3)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = nn.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = jt.matmul(attn_weights, v)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
        
        # Output projection
        output = self.out_proj(context)
        
        return output
        
        
# Create model factory function similar to the one in skeleton.py
def create_vqvae_model(model_type='standard', 
                      embedding_dim=512, 
                      num_embeddings=1024,
                      hidden_dims=None,
                      num_joints=52,
                      **kwargs):
    """
    Factory function to create the VQVAE model
    
    Args:
        model_type (str): Type of VQVAE model ('standard' or 'enhanced')
        embedding_dim (int): Dimension of the latent embedding
        num_embeddings (int): Size of the codebook
        hidden_dims (list): List of hidden dimensions
        num_joints (int): Number of joints to predict (22 or 52)
        **kwargs: Additional arguments for specific models
        
    Returns:
        VQVAE model
    """
    if hidden_dims is None:
        hidden_dims = [128, 256, 512]
        
    if model_type == 'standard':
        return VQVAE(
            point_dim=3,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            hidden_dims=hidden_dims,
            num_joints=num_joints,
            **kwargs
        )
    elif model_type == 'enhanced':
        return EnhancedVQVAE(
            point_dim=3,
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            hidden_dims=hidden_dims,
            num_joints=num_joints,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

