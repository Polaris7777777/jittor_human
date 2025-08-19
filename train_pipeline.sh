# CUDA_VISIBLE_DEVICES=5 python train_skeleton.py --train_data_list data/train_list.txt --val_data_list data/val_list.txt \
#         --data_root data --model_name sal --output_dir output/sal_t768_pc8192_w512_h8_e12_all_cos_emb_whnewnormals_normals_aug/skeleton \
#         --batch_size 8 --lr_step 20 --lr_decay 0.95  --epochs 400 \
#         --optimizer adamw --learning_rate 5e-6  --lr_scheduler cosine --lr_min 2e-7 \
#         --num_samples 16384 --vertex_samples 8192 \
#         --rotation_range 90.0 --scaling_range 0.8 1.2 \
#         --J2J_loss_weight 1.0 --sym_loss_weight 0.5 --bone_length_symmetry_weight 0.5 \
#         --topo_loss_weight 0.1 --rel_pos_loss_weight 0.1\
#         --use_normals_interior --mesh_interior_weight 0.5 --interior_margin 0.01 \
#         # --interior_k_neighbors 64 --use_advanced_interior \


# CUDA_VISIBLE_DEVICES=3 python train_skeleton.py --train_data_list data/train_list.txt --val_data_list data/val_list.txt \
#         --data_root data --model_name sal --output_dir output/ft_sal_t512_pc8192_w512_h8_all_cos_emb_whnormals_normals/skeleton \
#         --batch_size 16 --lr_step 20 --lr_decay 0.95  --epochs 200 \
#         --optimizer adamw --learning_rate 1e-6  --lr_scheduler cosine --lr_min 2e-7 \
#         --num_samples 8192 --vertex_samples 4096 \
#         --rotation_range 0.0 --scaling_range 1.0 1.0 \
#         --J2J_loss_weight 1.0 --sym_loss_weight 0.05 --bone_length_symmetry_weight 0.5 \
#         --topo_loss_weight 0.1 --rel_pos_loss_weight 0.1\
#         --use_normals_interior --mesh_interior_weight 0.5 --interior_margin 0.01 \
#         --pretrained_model output/ft_sal_t512_pc8192_w512_h8_all_cos_emb_whnewnormals_normals_laug/skeleton/best_model.pkl

# CUDA_VISIBLE_DEVICES=0 python train_skin.py --train_data_list data/train_list.txt --val_data_list data/val_list.txt \
#         --data_root data --model_name pct --output_dir output/pct_ns8192_f512_pf256_noise_aug/skin \
#         --batch_size 6 --lr_step 50  --lr_decay 0.9 \
#         --optimizer adamw --learning_rate 0.0005 \
#         --num_samples 8192 --vertex_samples 4096 \
#         --feat_dim 512 --pct_feat_dim 256 \
#         --enable_joint_noise --joint_noise_type adaptive --joint_noise_scale 0.02 --joint_noise_prob 0.5 \
#         --drop_bad --aug_prob 0.5 --rotation_range 90.0 --scaling_range 0.8 1.2 \


# CUDA_VISIBLE_DEVICES=3 python train_skeleton.py --train_data_list data/train_list.txt --val_data_list data/val_list.txt \
#         --data_root data --model_name sal --output_dir output/ft_sal_t768_pc8192_w512_h8_e12_all_cos_emb_whnormals_normals/skeleton \
#         --batch_size 8 --lr_step 20 --lr_decay 0.95  --epochs 200 \
#         --optimizer adamw --learning_rate 1e-6  --lr_scheduler cosine --lr_min 2e-7 \
#         --num_samples 8192 --vertex_samples 4096 \
#         --num_tokens 768 --feat_dim 512 --encoder_layers 12 --wnormals \
#         --drop_bad --aug_prob 0.0 --rotation_range 0.0 --scaling_range 1.0 1.0 \
#         --J2J_loss_weight 1.0 --sym_loss_weight 0.05 --bone_length_symmetry_weight 0.5 \
#         --topo_loss_weight 0.1 --rel_pos_loss_weight 0.1\
#         --use_normals_interior --mesh_interior_weight 0.5 --interior_margin 0.01 \
#         --pretrained_model output/sal_t768_pc8192_w512_h8_e12_all_cos_emb_whnewnormals_normals_aug/skeleton/best_model.pkl


# # -0804-
# CUDA_VISIBLE_DEVICES=2 python train_skeleton.py --train_data_list data/train_list.txt --val_data_list data/val_list.txt \
#         --data_root data --model_name sal --output_dir output/sal_t1024_pc16384_w512_h8_e12_all_cos_emb_whnormals_normals_paug/skeleton \
#         --batch_size 6 --lr_step 20 --lr_decay 0.95  --epochs 500 \
#         --optimizer adamw --learning_rate 1e-6  --lr_scheduler cosine --lr_min 2e-7 \
#         --num_samples 16384 --vertex_samples 8092 \
#         --num_tokens 1024 --feat_dim 512 --encoder_layers 12 --wnormals \
#         --drop_bad --aug_prob 0.5 --pose_angle_range 30.0 --rotation_range 90.0 --scaling_range 1.2 0.8 \
#         --J2J_loss_weight 1.0 --sym_loss_weight 0.01 --bone_length_symmetry_weight 0.5 \
#         --topo_loss_weight 0.1 --rel_pos_loss_weight 0.1\
#         --use_normals_interior --mesh_interior_weight 0.5 --interior_margin 0.01 \

# # -0816-
# HARD=1 \
# CUDA_VISIBLE_DEVICES=5 \
# python train_skeleton.py --train_data_list dataB/train_list.txt --val_data_list dataB/val_list.txt \
#         --data_root dataB --model_name sal --output_dir output/sal_t1024_pc16384_w512_h8_e12_all_cos_emb_whnormals_normals_paug_load_nval/skeleton \
#         --batch_size 6 --lr_step 20 --lr_decay 0.95  --epochs 500 \
#         --optimizer adamw --learning_rate 1e-6  --lr_scheduler cosine --lr_min 2e-7 \
#         --num_samples 16384 --vertex_samples 8092 --num_skeletons 52\
#         --num_tokens 1024 --feat_dim 512 --encoder_layers 12 --wnormals \
#         --aug_prob 0.5 --pose_angle_range 30.0 --rotation_range 90.0 --scaling_range 1.2 0.8 \
#         --track_pose_aug \
#         --J2J_loss_weight 1.0 --sym_loss_weight 0.0 --bone_length_symmetry_weight 0.5 \
#         --topo_loss_weight 0.1 --rel_pos_loss_weight 0.1\
#         --use_normals_interior --mesh_interior_weight 0.5 --interior_margin 0.01 \
#         --pretrained_model output_old/sal_t1024_pc16384_w512_h8_e12_all_cos_emb_whnormals_normals_paug/skeleton/best_model.pkl
# HARD=1 \
# CUDA_VISIBLE_DEVICES=4 \
# python train_skin.py --train_data_list dataB/train_list.txt --val_data_list dataB/val_list.txt \
#         --data_root dataB --model_name pct --output_dir output/pct_ns8192_f512_pf256_aug/skin \
#         --batch_size 4 --lr_step 50  --lr_decay 0.9 \
#         --optimizer adamw --learning_rate 0.0005 \
#         --num_samples 8192 --vertex_samples 4096 \
#         --feat_dim 512 --pct_feat_dim 256 \
#         --track_pose_aug --aug_prob 0.5 --rotation_range 90.0 --scaling_range 0.8 1.2 \
#         --pose_angle_range 30.0\
#         # --pretrained_model output_old/pct_ns8192_f512_pf256_noise_aug/skin/best_model.pkl

# -0820-
HARD=0 \
CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 \
python train_skeleton.py --train_data_list dataB/train_list.txt --val_data_list dataB/val_list.txt \
        --data_root dataB --model_name sal --output_dir output/sal_body_t1024_pc16384_w512_h8_e12_all2_cos_emb_aug3_load_mpi/skeleton \
        --batch_size 12 --lr_step 20 --lr_decay 0.95  --epochs 300 \
        --optimizer adamw --learning_rate 1e-6  --lr_scheduler cosine --lr_min 1e-7 \
        --num_samples 16384 --vertex_samples 8092 --num_skeletons 52\
        --num_tokens 1024 --feat_dim 512 --encoder_layers 12 --wnormals \
        --aug_prob 0.5 --pose_angle_range 10.0 --rotation_range 90.0 --scaling_range 0.5 1.5 \
        --track_pose_aug \
        --J2J_loss_weight 0.5 --sym_loss_weight 0.0 --bone_length_symmetry_weight 0.5 \
        --topo_loss_weight 0.5 --rel_pos_loss_weight 0.1\
        --use_normals_interior --mesh_interior_weight 1.0 --interior_margin 0.01 \
        --pretrained_model output_old/sal_t1024_pc16384_w512_h8_e12_all_cos_emb_whnormals_normals_paug/skeleton/best_model.pkl

# # -0818-
# HARD=1 \
# CUDA_VISIBLE_DEVICES=7 \
# python train_skeleton.py --train_data_list dataB/train_list.txt --val_data_list dataB/val_list.txt \
#         --data_root dataB --model_name sal --output_dir output/sal_t1024_pc16384_w512_h8_e12_all2_cos_emb_aug2_load/skeleton \
#         --batch_size 6 --lr_step 20 --lr_decay 0.95  --epochs 300 \
#         --optimizer adamw --learning_rate 1e-6  --lr_scheduler cosine --lr_min 1e-7 \
#         --num_samples 16384 --vertex_samples 8092 --num_skeletons 52\
#         --num_tokens 1024 --feat_dim 512 --encoder_layers 12 --wnormals \
#         --aug_prob 0.8 --pose_angle_range 10.0 --rotation_range 90.0 --scaling_range 1.5 0.5 \
#         --track_pose_aug \
#         --J2J_loss_weight 1.0 --sym_loss_weight 0.0 --bone_length_symmetry_weight 0.5 \
#         --topo_loss_weight 0.5 --rel_pos_loss_weight 0.1\
#         --use_normals_interior --mesh_interior_weight 1.0 --interior_margin 0.01 \
#         --pretrained_model output_old/sal_t1024_pc16384_w512_h8_e12_all_cos_emb_whnormals_normals_paug/skeleton/best_model.pkl
