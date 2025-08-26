#!/bin/bash

HARD=${HARD:-1}  # 默认使用B榜任务(52个骨骼点)

# 获取命令行参数
args="$@"

# 根据HARD环境变量设置数据集文件路径
if [ "$HARD" = "1" ]; then
    DATA_ROOT="dataB"
    echo "Using B-Board task with 52 joints (including hands)"
else
    DATA_ROOT="data"
    echo "Using A-Board task with 22 joints"
fi

# 运行训练脚本
CUDA_VISIBLE_DEVICES=0 \
python train_vqvae_skeleton.py \
    --train_data_list ${DATA_ROOT}/train_list.txt \
    --val_data_list ${DATA_ROOT}/val_list.txt \
    --data_root ${DATA_ROOT} \
    --model_type enhanced \
    --output_dir output/vqvae_enhanced_codebook1024 \
    --batch_size 8 \
    --epochs 300 \
    --optimizer adamw \
    --learning_rate 0.0001 \
    --lr_scheduler cosine \
    --lr_min 1e-6 \
    --num_samples 2048 \
    --vertex_samples 1024 \
    --embedding_dim 512 \
    --num_embeddings 1024 \
    --hidden_dims 128 256 512 \
    --use_transformer \
    --transformer_layers 4 \
    --transformer_heads 8 \
    --use_normals \
    --commitment_cost 0.25 \
    --aug_prob 0.5 \
    --rotation_range 90.0 \
    --scaling_range 0.8 1.2 \
    --pose_angle_range 30.0 \
    --track_pose_aug \
    --drop_bad \
    --vq_loss_weight 1.0 \
    --J2J_loss_weight 0.5 \
    --sym_loss_weight 0.05 \
    --bone_length_symmetry_weight 0.5 \
    --topo_loss_weight 0.1 \
    --rel_pos_loss_weight 0.1 \
    --use_mesh_interior \
    --mesh_interior_weight 0.5 \
    --interior_margin 0.01 \
    --use_normals_interior \
    --print_freq 10 \
    --save_freq 10 \
    --val_freq 1 \
    $args
