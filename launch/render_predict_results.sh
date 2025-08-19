predict_output_dir="predict/sal_t1024_pc16384_w512_h8_e12_all_cos_emb_whnormals_normals_paug_load_nval"
render_output_dir="render/sal_t1024_pc16384_w512_h8_e12_all_cos_emb_whnormals_normals_paug_load_nval"
render="false"
export_fbx="true"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --predict_output_dir) predict_output_dir="$2"; shift ;;
        --render_output_dir) render_output_dir="$2"; shift ;;
        --render) render="$2"; shift ;;
        --export_fbx) export_fbx="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

cmd=" \
    python render_predict_results.py \
    --predict_output_dir $predict_output_dir \
    --render_output_dir $render_output_dir \
    --render $render \
    --export_fbx $export_fbx \
"

cmd="$cmd &"
eval $cmd

wait