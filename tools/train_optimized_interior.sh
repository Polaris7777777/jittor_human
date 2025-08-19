#!/bin/bash

# 使用优化后的内部约束训练骨骼预测模型

echo "开始使用优化内部约束训练骨骼预测模型..."

# 基础配置
DATA_ROOT="data"
TRAIN_LIST="data/train_list.txt"
VAL_LIST="data/val_list.txt"
OUTPUT_DIR="output/skeleton_optimized_interior"
BATCH_SIZE=8
EPOCHS=100
LEARNING_RATE=0.0001

# 内部约束配置
MESH_INTERIOR_WEIGHT=1.0
INTERIOR_MARGIN=0.02
INTERIOR_K_NEIGHBORS=64
USE_ADVANCED_INTERIOR=true

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 运行训练
python train_skeleton.py \
    --train_data_list ${TRAIN_LIST} \
    --val_data_list ${VAL_LIST} \
    --data_root ${DATA_ROOT} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --optimizer adamw \
    --lr_scheduler cosine \
    --lr_min 1e-6 \
    --model_name sal \
    --num_samples 4096 \
    --vertex_samples 2048 \
    --mesh_interior_weight ${MESH_INTERIOR_WEIGHT} \
    --interior_margin ${INTERIOR_MARGIN} \
    --interior_k_neighbors ${INTERIOR_K_NEIGHBORS} \
    --use_advanced_interior \
    --interior_penalty_weight 2.0 \
    --sym_loss_weight 0.1 \
    --bone_length_symmetry_weight 0.1 \
    --J2J_loss_weight 0.1 \
    --topo_loss_weight 0.05 \
    --rel_pos_loss_weight 0.05 \
    --print_freq 5 \
    --save_freq 10 \
    --val_freq 2

echo "训练完成！"
echo "结果保存在: ${OUTPUT_DIR}"
echo ""
echo "使用以下命令查看训练日志:"
echo "tail -f ${OUTPUT_DIR}/training_log.txt"
echo ""
echo "使用以下命令可视化损失曲线:"
echo "python vis_skeleton_loss_curve.py --output_dir ${OUTPUT_DIR}"
