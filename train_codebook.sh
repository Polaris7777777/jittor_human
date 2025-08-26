#!/bin/bash

# Set environment variables

export HARD=1 # Set to 1 for 52 joints, 0 for 22 joints

# Training parameters
DATA_ROOT="dataB"
TRAIN_DATA_LIST="dataB/train_list.txt"
VAL_DATA_LIST="dataB/val_list.txt"
OUTPUT_DIR="output/keypoint_codebook_pose_aware_strong_aug_diversity_small2_reproduce_mpi"

# Model parameters (updated for Transformer)
EMBEDDING_DIM=32
NUM_EMBEDDINGS=128  # 增加codebook大小以处理更复杂的pose
HIDDEN_DIM=256
NUM_LAYERS=2
NUM_HEADS=8
COMMITMENT_COST=0.25

# Training parameters
BATCH_SIZE=96
EPOCHS=200
LEARNING_RATE=0.0001
OPTIMIZER="adamw"
LR_SCHEDULER="cosine"

# Enhanced data augmentation for B榜任务
ROTATION_RANGE=90.0  # 适度的旋转增强
SCALING_RANGE="0.5 1.5"  # 轻微的缩放增强
AUG_PROB=0.8  # 增加数据增强概率
POSE_ANGLE_RANGE=10.0  # 重要：姿态角度增强
TRACK_POSE_AUG="True"  # 使用动捕数据增强

# Loss weights - 调整权重以适应更复杂的关节结构
VQ_LOSS_WEIGHT=1.0
J2J_LOSS_WEIGHT=1.0  # 增加关节准确性权重
SYM_LOSS_WEIGHT=0.1  # 增加对称性约束
BONE_LENGTH_SYMMETRY_WEIGHT=0.8  # 重要：骨长对称性
TOPO_LOSS_WEIGHT=0.2  # 增加拓扑约束
REL_POS_LOSS_WEIGHT=0.15  # 相对位置约束

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training
CUDA_VISIBLE_DEVICES=6,7 mpirun -np 2 \
python train_codebook.py \
    --train_data_list $TRAIN_DATA_LIST \
    --val_data_list $VAL_DATA_LIST \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --model_type pose_aware_vqvae \
    --embedding_dim $EMBEDDING_DIM \
    --num_embeddings $NUM_EMBEDDINGS \
    --hidden_dim $HIDDEN_DIM \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --commitment_cost $COMMITMENT_COST \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --optimizer $OPTIMIZER \
    --lr_scheduler $LR_SCHEDULER \
    --rotation_range $ROTATION_RANGE \
    --scaling_range $SCALING_RANGE \
    --aug_prob $AUG_PROB \
    --pose_angle_range $POSE_ANGLE_RANGE \
    --track_pose_aug \
    --vq_loss_weight $VQ_LOSS_WEIGHT \
    --J2J_loss_weight $J2J_LOSS_WEIGHT \
    --sym_loss_weight $SYM_LOSS_WEIGHT \
    --bone_length_symmetry_weight $BONE_LENGTH_SYMMETRY_WEIGHT \
    --topo_loss_weight $TOPO_LOSS_WEIGHT \
    --rel_pos_loss_weight $REL_POS_LOSS_WEIGHT \
    --print_freq 10 \
    --save_freq 10 \
    --val_freq 1

echo "Training completed!"
