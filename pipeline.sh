# # =======================================Skeleton Model============================================
# # ----------------------------------------step 1 train----------------------------------------
# CUDA_VISIBLE_DEVICES=0 python train_skeleton.py --train_data_list data/train_list.txt --val_data_list data/val_list.txt \
#         --data_root data --model_name sal --output_dir output/sal_t512_pc8192_w512_h8_e8_all_cos_emb_whnormals_normals_aug/skeleton \
#         --batch_size 8 --lr_step 20 --lr_decay 0.95  --epochs 500 \
#         --optimizer adamw --learning_rate 1e-5  --lr_scheduler cosine --lr_min 2e-7 \
#         --num_samples 8192 --vertex_samples 4096 \
#         --num_tokens 512 --feat_dim 512 --encoder_layers 8 \
#         --aug_prob 1.0 --rotation_range 90.0 --scaling_range 0.8 1.2 \
#         --J2J_loss_weight 1.0 --sym_loss_weight 0.5 --bone_length_symmetry_weight 0.5 \
#         --topo_loss_weight 0.1 --rel_pos_loss_weight 0.1\
#         --use_normals_interior --mesh_interior_weight 0.5 --interior_margin 0.01 \
#         --terminal_interior_loss
# # ----------------------------------------step 2 ft w/ laug----------------------------------------
# CUDA_VISIBLE_DEVICES=0 python train_skeleton.py --train_data_list data/train_list.txt --val_data_list data/val_list.txt \
#         --data_root data --model_name sal --output_dir output/ft_sal_t512_pc8192_w512_h8_e8_all_cos_emb_whnormals_normals_laug/skeleton \
#         --batch_size 8 --lr_step 20 --lr_decay 0.95  --epochs 200 \
#         --optimizer adamw --learning_rate 5e-6  --lr_scheduler cosine --lr_min 2e-7 \
#         --num_samples 8192 --vertex_samples 4096 \
#         --num_tokens 512 --feat_dim 512 --encoder_layers 8 --wnormals\
#         --aug_prob 0.5 --rotation_range 90.0 --scaling_range 0.8 1.2 \
#         --J2J_loss_weight 1.0 --sym_loss_weight 0.5 --bone_length_symmetry_weight 0.5 \
#         --topo_loss_weight 0.1 --rel_pos_loss_weight 0.1\
#         --use_normals_interior --mesh_interior_weight 0.5 --interior_margin 0.01 \
#         --pretrained_model output/sal_t512_pc8192_w512_h8_e8_all_cos_emb_whnormals_normals_aug/skeleton/best_model.pkl \

# # ----------------------------------------step 3 ft w/o aug + drop bad----------------------------------------
# CUDA_VISIBLE_DEVICES=0 python train_skeleton.py --train_data_list data/train_list.txt --val_data_list data/val_list.txt \
#         --data_root data --model_name sal --output_dir output/ft_sal_t512_pc8192_w512_h8_e8_all_cos_emb_whnormals_normals/skeleton \
#         --batch_size 8 --lr_step 20 --lr_decay 0.95  --epochs 200 \
#         --optimizer adamw --learning_rate 5e-6  --lr_scheduler cosine --lr_min 2e-7 \
#         --num_samples 8192 --vertex_samples 4096 \
#         --num_tokens 512 --feat_dim 512 --encoder_layers 8 --wnormals\
#         --aug_prob 0.0 --rotation_range 0.0 --scaling_range 1.0 1.0 --drop_bad \
#         --J2J_loss_weight 1.0 --sym_loss_weight 0.5 --bone_length_symmetry_weight 0.5 \
#         --topo_loss_weight 0.1 --rel_pos_loss_weight 0.1\
#         --use_normals_interior --mesh_interior_weight 0.5 --interior_margin 0.01 \
#         --pretrained_model output/ft_sal_t512_pc8192_w512_h8_e8_all_cos_emb_whnormals_normals_laug/skeleton/best_model.pkl \

# ==================================================Skeleton Model========================================================
CUDA_VISIBLE_DEVICES=0 python train_skeleton.py --train_data_list data/train_list.txt --val_data_list data/val_list.txt \
        --data_root data --model_name sal --output_dir output/sal_t768_pc16384_w512_h8_e12_all_cos_emb_whnormals_normals_paug/skeleton \
        --batch_size 6 --lr_step 20 --lr_decay 0.95  --epochs 500 \
        --optimizer adamw --learning_rate 1e-6  --lr_scheduler cosine --lr_min 2e-7 \
        --num_samples 16384 --vertex_samples 8092 \
        --num_tokens 768 --feat_dim 512 --encoder_layers 12 --wnormals \
        --drop_bad --aug_prob 0.5 --pose_angle_range 0.0 --rotation_range 90.0 --scaling_range 1.2 0.8 \
        --J2J_loss_weight 1.0 --sym_loss_weight 0.05 --bone_length_symmetry_weight 0.5 \
        --topo_loss_weight 0.1 --rel_pos_loss_weight 0.1\
        --use_normals_interior --mesh_interior_weight 0.5 --interior_margin 0.01 \

# ===================================================Skin Model========================================================
CUDA_VISIBLE_DEVICES=0 python train_skin.py --train_data_list data/train_list.txt --val_data_list data/val_list.txt \
        --data_root data --model_name pct --output_dir output/pct_ns8192_f256_pf256/skin \
        --batch_size 6 --lr_step 50  --lr_decay 0.9 \
        --optimizer adamw --learning_rate 0.0005 \
        --num_samples 8192 --vertex_samples 4096 \
        --feat_dim 256 --pct_feat_dim 256 \
