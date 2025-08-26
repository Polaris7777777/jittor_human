```bash
# Model parameters - 参考PCT的参数设置，大幅增加以防止collapse
EMBEDDING_DIM=512       # 从32增加到512，提供足够的表达能力
NUM_EMBEDDINGS=2048     # 从64大幅增加到2048，PCT使用的规模
HIDDEN_DIM=512          # 增加到512
NUM_LAYERS=4            # 增加层数
NUM_HEADS=8
COMMITMENT_COST=0.25    # 降低到PCT的标准值
DROPOUT=0.1             # 降低dropout
WEIGHT_DECAY=1e-4       # 适中的正则化

# Training parameters - 调整训练策略
BATCH_SIZE=16           # 减小batch size，增加更新频率
EPOCHS=100              # 减少epoch，先看效果
LEARNING_RATE=0.001     # 提高学习率
OPTIMIZER="adamw"
LR_SCHEDULER="cosine"

# Enhanced data augmentation
ROTATION_RANGE=180.0    
SCALING_RANGE="0.7 1.3"  
AUG_PROB=0.9            # 几乎总是增强
POSE_ANGLE_RANGE=30.0   
TRACK_POSE_AUG="True"   

# Loss weights - 大幅降低VQ权重，增加多样性
VQ_LOSS_WEIGHT=0.1      # 大幅降低VQ损失权重
J2J_LOSS_WEIGHT=10.0    # 大幅增加重建质量权重
```