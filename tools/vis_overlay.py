import cv2
import numpy as np
import os
import argparse
from PIL import Image

def overlay_skeleton_on_skin(skeleton_path, skin_path, output_path, alpha=0.7):
    """
    将skeleton图像叠加到skin图像上
    
    Args:
        skeleton_path: skeleton图像路径
        skin_path: skin图像路径  
        output_path: 输出图像路径
        alpha: skin图像的透明度 (0-1)
    """
    # 读取图像
    skeleton_img = cv2.imread(skeleton_path)
    skin_img = cv2.imread(skin_path)
    
    if skeleton_img is None or skin_img is None:
        print(f"Error loading images: {skeleton_path} or {skin_path}")
        return
    
    # 确保两个图像尺寸相同
    if skeleton_img.shape != skin_img.shape:
        skin_img = cv2.resize(skin_img, (skeleton_img.shape[1], skeleton_img.shape[0]))
    
    # 叠加图像：alpha*skin + (1-alpha)*skeleton
    overlay_img = cv2.addWeighted(skin_img, alpha, skeleton_img, 1-alpha, 0)
    
    # 保存结果
    cv2.imwrite(output_path, overlay_img)

def process_single_directory(input_dir, output_dir, alpha=0.7):
    """
    处理单个目录，将skeleton叠加到所有skin图像上
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        alpha: skin图像的透明度
    """
    skeleton_path = os.path.join(input_dir, 'skeleton.png')
    
    if not os.path.exists(skeleton_path):
        print(f"Skeleton image not found: {skeleton_path}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理所有skin_*.png文件
    for filename in os.listdir(input_dir):
        if filename.startswith('skin_') and filename.endswith('.png'):
            skin_path = os.path.join(input_dir, filename)
            # 生成叠加后的文件名，如 skin_chest.png -> overlay_chest.png
            overlay_filename = filename.replace('skin_', 'overlay_')
            output_path = os.path.join(output_dir, overlay_filename)
            overlay_skeleton_on_skin(skeleton_path, skin_path, output_path, alpha)
    
    # 处理sampled_vertices.png（如果存在）
    sampled_vertices_path = os.path.join(input_dir, 'sampled_vertices.png')
    if os.path.exists(sampled_vertices_path):
        output_path = os.path.join(output_dir, 'overlay_sampled_vertices.png')
        overlay_skeleton_on_skin(skeleton_path, sampled_vertices_path, output_path, alpha)

def process_render_results(input_root_dir, output_root_dir, alpha=0.7):
    """
    处理整个渲染结果目录结构（支持嵌套目录）
    
    Args:
        input_root_dir: 输入根目录路径
        output_root_dir: 输出根目录路径
        alpha: skin图像的透明度
    """
    print(f"Processing render results from {input_root_dir} to {output_root_dir}")
    
    # 递归遍历所有目录，查找包含skeleton.png的目录
    for root, dirs, files in os.walk(input_root_dir):
        # 检查当前目录是否包含skeleton.png和skin文件
        if 'skeleton.png' in files:
            has_skin_files = any(f.startswith('skin_') and f.endswith('.png') for f in files)
            
            if has_skin_files:
                # 计算相对路径
                rel_path = os.path.relpath(root, input_root_dir)
                print(f"Processing directory: {rel_path}")
                
                # 创建对应的输出目录
                output_dir = os.path.join(output_root_dir, rel_path)
                process_single_directory(root, output_dir, alpha)
            else:
                rel_path = os.path.relpath(root, input_root_dir)
                print(f"Skipping directory {rel_path}: has skeleton.png but no skin files")

def create_overlays_for_directory(save_path, id_to_name):
    """
    为指定目录创建所有skin与skeleton的叠加图像
    
    Args:
        save_path: 保存路径
        id_to_name: 关节ID到名称的映射
    """
    skeleton_path = os.path.join(save_path, 'skeleton.png')
    
    if not os.path.exists(skeleton_path):
        print(f"Skeleton image not found: {skeleton_path}")
        return
    
    # 创建overlay子目录
    overlay_dir = os.path.join(save_path, 'overlay')
    os.makedirs(overlay_dir, exist_ok=True)
    
    # 为每个关节部位创建叠加图像
    for id in id_to_name:
        name = id_to_name[id]
        skin_path = os.path.join(save_path, f'skin_{name}.png')
        
        if os.path.exists(skin_path):
            output_path = os.path.join(overlay_dir, f'overlay_{name}.png')
            overlay_skeleton_on_skin(skeleton_path, skin_path, output_path)
    
    # 如果存在sampled_vertices图像，也创建叠加
    sampled_vertices_path = os.path.join(save_path, 'sampled_vertices.png')
    if os.path.exists(sampled_vertices_path):
        output_path = os.path.join(overlay_dir, 'overlay_sampled_vertices.png')
        overlay_skeleton_on_skin(skeleton_path, sampled_vertices_path, output_path)

def main():
    parser = argparse.ArgumentParser(description='将skeleton叠加到skin渲染结果上')
    parser.add_argument('--input_dir', type=str, default="render/sal_t768_pc16384_w512_h8_e12_all_cos_emb_whnormals_normals_aug",
                       help='输入目录路径（包含渲染结果的根目录）')
    parser.add_argument('--output_dir', type=str, default="render/sal_t768_pc16384_w512_h8_e12_all_cos_emb_whnormals_normals_aug_overlay",
                       help='输出目录路径')
    parser.add_argument('--alpha', type=float, default=0.7,
                       help='skin图像的透明度 (0-1), 默认0.7')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"输入目录不存在: {args.input_dir}")
        return
    
    process_render_results(args.input_dir, args.output_dir, args.alpha)
    print("处理完成！")

if __name__ == "__main__":
    main()
