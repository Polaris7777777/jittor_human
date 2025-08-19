# CUDA_VISIBLE_DEVICES=1 python predict_skeleton.py --predict_data_list data/test_list.txt --data_root data \
#        --pretrained_model output/ft_sal_t512_pc8192_w512_h8_all_cos_emb_whnormals_normals/skeleton/best_model.pkl \
#        --predict_output_dir predict/sal_t512_pc8192_w512_h8_all_cos_emb_whnormals_normals_ft \
#        --model_name sal --wnormals \
#        --batch_size 2 --num_samples 8192 --vertex_samples 4096 \
# ----------------------------------------List A submit----------------------------------------


# -------------------------------------------------------------------------------------------------
# CUDA_VISIBLE_DEVICES=7 python predict_skeleton.py --predict_data_list data/test_list.txt --data_root data \
#        --pretrained_model output/sal_t768_pc16384_w512_h8_e12_all_cos_emb_whnormals_normals_aug/skeleton/best_model.pkl \
#        --predict_output_dir predict/sal_t768_pc16384_w512_h8_e12_all_cos_emb_whnormals_normals_aug_re_eval_new \
#        --model_name sal --wnormals  --num_tokens 768 --feat_dim 512 --encoder_layers 12 \
#        --batch_size 2 --num_samples 16384 --vertex_samples 8192 \
# ----------------------------------------New List A submit----------------------------------------

# CUDA_VISIBLE_DEVICES=7 python predict_skeleton.py --predict_data_list data/test_list.txt --data_root data \
#        --pretrained_model output/sal_t1024_pc16384_w512_h8_e12_all_cos_emb_whnormals_normals_aug/skeleton/best_model.pkl \
#        --predict_output_dir predict/sal_t1024_pc16384_w512_h8_e12_all_cos_emb_whnormals_normals_aug \
#        --model_name sal --wnormals  --num_tokens 1024 --feat_dim 512 --encoder_layers 12 \
#        --batch_size 2 --num_samples 16384 --vertex_samples 8192 \

# # 0817 submit
# CUDA_VISIBLE_DEVICES=7 python predict_skeleton.py --predict_data_list dataB/test_list.txt --data_root dataB \
#        --pretrained_model output/sal_t1024_pc16384_w512_h8_e12_all_cos_emb_whnormals_normals_paug_load/skeleton/best_model.pkl \
#        --predict_output_dir predict/sal_t1024_pc16384_w512_h8_e12_all_cos_emb_whnormals_normals_paug_load \
#        --model_name sal --wnormals  --num_tokens 1024 --feat_dim 512 --encoder_layers 12  \
#        --batch_size 2 --num_samples 16384 --vertex_samples 8192 \

# 0817 submit
CUDA_VISIBLE_DEVICES=7 python predict_skeleton.py --predict_data_list dataB/test_list.txt --data_root dataB \
       --pretrained_model output/sal_t1024_pc16384_w512_h8_e12_all_cos_emb_whnormals_normals_paug_load_nval/skeleton/best_model.pkl \
       --predict_output_dir predict/sal_t1024_pc16384_w512_h8_e12_all_cos_emb_whnormals_normals_paug_load_nval \
       --model_name sal --wnormals  --num_tokens 1024 --feat_dim 512 --encoder_layers 12  \
       --batch_size 2 --num_samples 16384 --vertex_samples 8192 \