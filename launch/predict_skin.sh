# CUDA_VISIBLE_DEVICES=0 python predict_skin.py --predict_data_list data/test_list.txt --data_root data --model_name pct \
#           --pretrained_model output/pct_ns8192_f256_pf256/skin/best_model.pkl \
#           --predict_output_dir predict/sal_t512_pc8192_w512_h8_all_cos_emb_whnormals_normals_ft \
#           --batch_size 4 --num_samples 8192 --vertex_samples 4096 \
#           --feat_dim 256 --pct_feat_dim 256
# -------------------------------------List A submit----------------------------------------


# --------------------------------------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=7 python predict_skin.py --predict_data_list data/test_list.txt --data_root data --model_name pct \
#           --pretrained_model output/pct_ns8192_f256_pf256/skin/best_model.pkl \
#           --predict_output_dir predict/sal_t768_pc16384_w512_h8_e12_all_cos_emb_whnormals_normals_aug_re_eval_new \
#           --batch_size 4 --num_samples 8192 --vertex_samples 4096 \
#           --feat_dim 256 --pct_feat_dim 256
# -------------------------------------New List A submit----------------------------------------

# CUDA_VISIBLE_DEVICES=7 python predict_skin.py --predict_data_list data/test_list.txt --data_root data --model_name pct \
#           --pretrained_model output/pct_ns4096_f256_pf256_noise/skin/best_model.pkl \
#           --predict_output_dir predict/ft_sal_t768_pc8192_w512_h8_e12_all_cos_emb_whnormals_normals \
#           --batch_size 4 --num_samples 8192 --vertex_samples 4096 \
#           --feat_dim 256 --pct_feat_dim 256
          

# CUDA_VISIBLE_DEVICES=0 python predict_skin.py --predict_data_list data/test_list.txt --data_root data --model_name pct \
#           --pretrained_model output/pct_ds2_0723/skin/best_model.pkl \
#           --predict_output_dir predict/sal_t512_pc8192_w512_h8_all_cos_emb_wnormals_normals_2 \
#           --batch_size 8 --num_samples 2048 --vertex_samples 1024 \
#           --feat_dim 256 --pct_feat_dim 128

# # 0817 submit
# CUDA_VISIBLE_DEVICES=7 python predict_skin.py --predict_data_list dataB/test_list.txt --data_root dataB --model_name pct \
#           --pretrained_model output/pct_ns8192_f512_pf256_aug_load/skin/best_model.pkl \
#           --predict_output_dir predict/sal_t1024_pc16384_w512_h8_e12_all_cos_emb_whnormals_normals_paug_load \
#           --batch_size 1 --num_samples 8192 --vertex_samples 4096 \
#           --feat_dim 512 --pct_feat_dim 256

# 0817 submit
CUDA_VISIBLE_DEVICES=7 python predict_skin.py --predict_data_list dataB/test_list.txt --data_root dataB --model_name pct \
          --pretrained_model output/pct_ns8192_f512_pf256_aug_load/skin/best_model.pkl \
          --predict_output_dir predict/sal_t1024_pc16384_w512_h8_e12_all_cos_emb_whnormals_normals_paug_load_nval \
          --batch_size 1 --num_samples 8192 --vertex_samples 4096 \
          --feat_dim 512 --pct_feat_dim 256