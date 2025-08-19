#!/bin/bash

# 使用Mesh内部约束的骨骼训练脚本示例
# 这个脚本展示了如何使用新添加的mesh_interior_loss来训练骨骼预测模型

echo "开始使用Mesh内部约束的骨骼训练..."

# 设置基本参数
DATA_ROOT="data"
TRAIN_LIST="data/train_list.txt"
VAL_LIST="data/val_list.txt"
OUTPUT_DIR="output/skeleton_with_interior_constraint"
MODEL_NAME="sal"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 训练参数
BATCH_SIZE=8
EPOCHS=200
LEARNING_RATE=0.00005
NUM_SAMPLES=2048
VERTEX_SAMPLES=1024

# 损失权重参数
MSE_WEIGHT=1.0                    # 基础MSE损失权重
SYM_WEIGHT=0.1                    # 对称性损失权重
BONE_SYM_WEIGHT=0.1               # 骨骼长度对称性损失权重
J2J_WEIGHT=0.1                    # J2J损失权重
TOPO_WEIGHT=0.1                   # 拓扑损失权重
REL_POS_WEIGHT=0.1                # 相对位置损失权重
INTERIOR_WEIGHT=0.5               # 🆕 Mesh内部约束损失权重

# 内部约束参数
INTERIOR_MARGIN=0.01              # 内部边界安全边距
INTERIOR_K_NEIGHBORS=50           # 用于计算内部约束的邻居数量

echo "训练配置:"
echo "  数据根目录: $DATA_ROOT"
echo "  训练列表: $TRAIN_LIST"
echo "  验证列表: $VAL_LIST"
echo "  输出目录: $OUTPUT_DIR"
echo "  模型类型: $MODEL_NAME"
echo "  批大小: $BATCH_SIZE"
echo "  训练轮数: $EPOCHS"
echo "  学习率: $LEARNING_RATE"
echo "  点云采样数: $NUM_SAMPLES"
echo "  顶点采样数: $VERTEX_SAMPLES"
echo ""
echo "损失权重:"
echo "  MSE权重: $MSE_WEIGHT"
echo "  对称性权重: $SYM_WEIGHT"
echo "  骨骼对称性权重: $BONE_SYM_WEIGHT"
echo "  J2J权重: $J2J_WEIGHT"
echo "  拓扑权重: $TOPO_WEIGHT"
echo "  相对位置权重: $REL_POS_WEIGHT"
echo "  🆕 内部约束权重: $INTERIOR_WEIGHT"
echo ""
echo "内部约束参数:"
echo "  安全边距: $INTERIOR_MARGIN"
echo "  邻居数量: $INTERIOR_K_NEIGHBORS"
echo ""

# 运行训练
python train_skeleton.py \
    --train_data_list $TRAIN_LIST \
    --val_data_list $VAL_LIST \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --model_name $MODEL_NAME \
    --model_type "standard" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --optimizer "adamw" \
    --learning_rate $LEARNING_RATE \
    --weight_decay 1e-4 \
    --lr_scheduler "cosine" \
    --lr_min 1e-6 \
    --num_samples $NUM_SAMPLES \
    --vertex_samples $VERTEX_SAMPLES \
    --sym_loss_weight $SYM_WEIGHT \
    --bone_length_symmetry_weight $BONE_SYM_WEIGHT \
    --J2J_loss_weight $J2J_WEIGHT \
    --topo_loss_weight $TOPO_WEIGHT \
    --rel_pos_loss_weight $REL_POS_WEIGHT \
    --mesh_interior_weight $INTERIOR_WEIGHT \
    --interior_margin $INTERIOR_MARGIN \
    --interior_k_neighbors $INTERIOR_K_NEIGHBORS \
    --print_freq 10 \
    --save_freq 10 \
    --val_freq 1

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 训练完成！"
    echo "输出文件位于: $OUTPUT_DIR"
    echo ""
    echo "训练产生的文件:"
    echo "  - best_model.pkl: 最佳模型"
    echo "  - final_model.pkl: 最终模型" 
    echo "  - training_log.txt: 训练日志"
    echo "  - checkpoint_epoch_*.pkl: 周期性检查点"
    echo ""
    echo "可以使用 vis_loss_curve.py 来可视化训练损失曲线"
    echo "可以使用 example_mesh_interior_loss.py 来测试内部约束损失"
else
    echo ""
    echo "❌ 训练失败，请检查错误信息"
    exit 1
fi
