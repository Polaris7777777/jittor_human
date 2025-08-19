import numpy as np
import os
import argparse
from tqdm import tqdm
import json

def calculate_difference(pred_dir1, pred_dir2, output_dir):
    """
    计算两次预测结果的差值
    
    Args:
        pred_dir1: 第一次预测结果目录
        pred_dir2: 第二次预测结果目录  
        output_dir: 差值结果输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有类别目录
    classes = os.listdir(pred_dir1)
    
    skeleton_diffs = []
    skin_diffs = []
    
    print("计算预测结果差值...")
    
    for cls in tqdm(classes):
        cls_dir1 = os.path.join(pred_dir1, cls)
        cls_dir2 = os.path.join(pred_dir2, cls)
        
        if not os.path.exists(cls_dir2):
            print(f"警告: {cls_dir2} 不存在，跳过类别 {cls}")
            continue
            
        # 获取该类别下的所有ID
        ids = os.listdir(cls_dir1)
        
        for id_name in ids:
            id_dir1 = os.path.join(cls_dir1, id_name)
            id_dir2 = os.path.join(cls_dir2, id_name)
            
            if not os.path.exists(id_dir2):
                print(f"警告: {id_dir2} 不存在，跳过ID {id_name}")
                continue
                
            # 创建输出目录
            output_cls_dir = os.path.join(output_dir, cls, id_name)
            os.makedirs(output_cls_dir, exist_ok=True)
            
            # 比较skeleton预测结果
            skeleton_file1 = os.path.join(id_dir1, "predict_skeleton.npy")
            skeleton_file2 = os.path.join(id_dir2, "predict_skeleton.npy")
            
            if os.path.exists(skeleton_file1) and os.path.exists(skeleton_file2):
                skeleton1 = np.load(skeleton_file1)
                skeleton2 = np.load(skeleton_file2)
                
                if skeleton1.shape == skeleton2.shape:
                    # 计算多种差值指标
                    skeleton_diff = skeleton1 - skeleton2
                    skeleton_diff_norm = np.linalg.norm(skeleton_diff, axis=-1)
                    
                    # MSE loss (类似训练代码)
                    mse_loss = np.mean((skeleton1 - skeleton2) ** 2)
                    
                    # L1 loss
                    l1_loss = np.mean(np.abs(skeleton1 - skeleton2))
                    
                    # 关节间距离差异 (joint-to-joint distance)
                    j2j_distance = np.mean(skeleton_diff_norm)
                    
                    # 最大关节位移
                    max_joint_displacement = np.max(skeleton_diff_norm)
                    
                    # 保存差值
                    np.save(os.path.join(output_cls_dir, "skeleton_diff.npy"), skeleton_diff)
                    np.save(os.path.join(output_cls_dir, "skeleton_diff_norm.npy"), skeleton_diff_norm)
                    
                    # 统计信息
                    skeleton_diffs.append({
                        'cls': cls,
                        'id': id_name,
                        'mse_loss': float(mse_loss),
                        'l1_loss': float(l1_loss),
                        'j2j_distance': float(j2j_distance),
                        'max_displacement': float(max_joint_displacement),
                        'mean_diff': float(np.mean(skeleton_diff_norm)),
                        'max_diff': float(np.max(skeleton_diff_norm)),
                        'min_diff': float(np.min(skeleton_diff_norm)),
                        'std_diff': float(np.std(skeleton_diff_norm))
                    })
                else:
                    print(f"警告: skeleton形状不匹配 {cls}/{id_name}: {skeleton1.shape} vs {skeleton2.shape}")
            
            # 比较skin预测结果
            skin_file1 = os.path.join(id_dir1, "predict_skin.npy")
            skin_file2 = os.path.join(id_dir2, "predict_skin.npy")
            
            if os.path.exists(skin_file1) and os.path.exists(skin_file2):
                skin1 = np.load(skin_file1)
                skin2 = np.load(skin_file2)
                
                if skin1.shape == skin2.shape:
                    # 计算多种差值指标
                    skin_diff = skin1 - skin2
                    skin_diff_norm = np.linalg.norm(skin_diff, axis=-1)
                    
                    # MSE loss (类似训练代码)
                    mse_loss = np.mean((skin1 - skin2) ** 2)
                    
                    # L1 loss
                    l1_loss = np.mean(np.abs(skin1 - skin2))
                    
                    # 皮肤权重的最大差异
                    max_weight_diff = np.max(np.abs(skin_diff))
                    
                    # 皮肤权重分布的KL散度或JS散度
                    # 确保权重和为1并且非负
                    skin1_normalized = np.abs(skin1) / (np.sum(np.abs(skin1), axis=1, keepdims=True) + 1e-8)
                    skin2_normalized = np.abs(skin2) / (np.sum(np.abs(skin2), axis=1, keepdims=True) + 1e-8)
                    
                    # JS散度计算
                    m = 0.5 * (skin1_normalized + skin2_normalized)
                    kl1 = np.sum(skin1_normalized * np.log((skin1_normalized + 1e-8) / (m + 1e-8)), axis=1)
                    kl2 = np.sum(skin2_normalized * np.log((skin2_normalized + 1e-8) / (m + 1e-8)), axis=1)
                    js_divergence = 0.5 * (kl1 + kl2)
                    mean_js_divergence = np.mean(js_divergence)
                    
                    # 保存差值
                    np.save(os.path.join(output_cls_dir, "skin_diff.npy"), skin_diff)
                    np.save(os.path.join(output_cls_dir, "skin_diff_norm.npy"), skin_diff_norm)
                    
                    # 统计信息
                    skin_diffs.append({
                        'cls': cls,
                        'id': id_name,
                        'mse_loss': float(mse_loss),
                        'l1_loss': float(l1_loss),
                        'max_weight_diff': float(max_weight_diff),
                        'js_divergence': float(mean_js_divergence),
                        'mean_diff': float(np.mean(skin_diff_norm)),
                        'max_diff': float(np.max(skin_diff_norm)),
                        'min_diff': float(np.min(skin_diff_norm)),
                        'std_diff': float(np.std(skin_diff_norm))
                    })
                else:
                    print(f"警告: skin形状不匹配 {cls}/{id_name}: {skin1.shape} vs {skin2.shape}")
    
    # 保存统计结果
    stats = {
        'skeleton_stats': {
            'count': len(skeleton_diffs),
            'overall_mse_loss': float(np.mean([item['mse_loss'] for item in skeleton_diffs])) if skeleton_diffs else 0,
            'overall_l1_loss': float(np.mean([item['l1_loss'] for item in skeleton_diffs])) if skeleton_diffs else 0,
            'overall_j2j_distance': float(np.mean([item['j2j_distance'] for item in skeleton_diffs])) if skeleton_diffs else 0,
            'overall_max_displacement': float(np.max([item['max_displacement'] for item in skeleton_diffs])) if skeleton_diffs else 0,
            'overall_mean_diff': float(np.mean([item['mean_diff'] for item in skeleton_diffs])) if skeleton_diffs else 0,
            'overall_max_diff': float(np.max([item['max_diff'] for item in skeleton_diffs])) if skeleton_diffs else 0,
            'details': skeleton_diffs
        },
        'skin_stats': {
            'count': len(skin_diffs),
            'overall_mse_loss': float(np.mean([item['mse_loss'] for item in skin_diffs])) if skin_diffs else 0,
            'overall_l1_loss': float(np.mean([item['l1_loss'] for item in skin_diffs])) if skin_diffs else 0,
            'overall_max_weight_diff': float(np.max([item['max_weight_diff'] for item in skin_diffs])) if skin_diffs else 0,
            'overall_js_divergence': float(np.mean([item['js_divergence'] for item in skin_diffs])) if skin_diffs else 0,
            'overall_mean_diff': float(np.mean([item['mean_diff'] for item in skin_diffs])) if skin_diffs else 0,
            'overall_max_diff': float(np.max([item['max_diff'] for item in skin_diffs])) if skin_diffs else 0,
            'details': skin_diffs
        }
    }
    
    with open(os.path.join(output_dir, "difference_stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    # 打印汇总统计
    print("\n=== 差值统计汇总 ===")
    print(f"Skeleton预测:")
    print(f"  样本数量: {stats['skeleton_stats']['count']}")
    print(f"  MSE Loss: {stats['skeleton_stats']['overall_mse_loss']:.6f}")
    print(f"  L1 Loss: {stats['skeleton_stats']['overall_l1_loss']:.6f}")
    print(f"  J2J距离: {stats['skeleton_stats']['overall_j2j_distance']:.6f}")
    print(f"  最大位移: {stats['skeleton_stats']['overall_max_displacement']:.6f}")
    
    print(f"\nSkin预测:")
    print(f"  样本数量: {stats['skin_stats']['count']}")
    print(f"  MSE Loss: {stats['skin_stats']['overall_mse_loss']:.6f}")
    print(f"  L1 Loss: {stats['skin_stats']['overall_l1_loss']:.6f}")
    print(f"  最大权重差异: {stats['skin_stats']['overall_max_weight_diff']:.6f}")
    print(f"  JS散度: {stats['skin_stats']['overall_js_divergence']:.6f}")
    
    # 分析异常值
    print("\n=== 异常值分析 ===")
    
    # Skeleton异常值分析
    if skeleton_diffs:
        skeleton_max_displacements = [item['max_displacement'] for item in skeleton_diffs]
        skeleton_mean = np.mean(skeleton_max_displacements)
        skeleton_std = np.std(skeleton_max_displacements)
        skeleton_threshold = skeleton_mean + 2 * skeleton_std
        
        skeleton_outliers = [item for item in skeleton_diffs if item['max_displacement'] > skeleton_threshold]
        print(f"Skeleton异常样本 (位移 > {skeleton_threshold:.3f}):")
        for outlier in skeleton_outliers[:5]:  # 显示前5个
            print(f"  {outlier['cls']}/{outlier['id']}: 最大位移={outlier['max_displacement']:.3f}, MSE={outlier['mse_loss']:.6f}")
    
    # Skin异常值分析
    if skin_diffs:
        skin_max_weights = [item['max_weight_diff'] for item in skin_diffs]
        skin_mean = np.mean(skin_max_weights)
        skin_std = np.std(skin_max_weights)
        skin_threshold = skin_mean + 2 * skin_std
        
        skin_outliers = [item for item in skin_diffs if item['max_weight_diff'] > skin_threshold]
        print(f"Skin异常样本 (权重差异 > {skin_threshold:.3f}):")
        for outlier in skin_outliers[:5]:  # 显示前5个
            print(f"  {outlier['cls']}/{outlier['id']}: 最大权重差异={outlier['max_weight_diff']:.3f}, JS散度={outlier['js_divergence']:.6f}")
    
    # 添加分布统计
    print("\n=== 分布统计 ===")
    if skeleton_diffs:
        skeleton_j2j_distances = [item['j2j_distance'] for item in skeleton_diffs]
        print(f"Skeleton J2J距离分布:")
        print(f"  中位数: {np.median(skeleton_j2j_distances):.6f}")
        print(f"  75%分位数: {np.percentile(skeleton_j2j_distances, 75):.6f}")
        print(f"  95%分位数: {np.percentile(skeleton_j2j_distances, 95):.6f}")
    
    if skin_diffs:
        skin_js_divergences = [item['js_divergence'] for item in skin_diffs]
        print(f"Skin JS散度分布:")
        print(f"  中位数: {np.median(skin_js_divergences):.6f}")
        print(f"  75%分位数: {np.percentile(skin_js_divergences, 75):.6f}")
        print(f"  95%分位数: {np.percentile(skin_js_divergences, 95):.6f}")
    
    # 添加数据源分析
    print("\n=== 数据源差异分析 ===")
    
    # 按数据集分组分析
    skeleton_by_dataset = {}
    skin_by_dataset = {}
    
    for item in skeleton_diffs:
        dataset = item['cls']
        if dataset not in skeleton_by_dataset:
            skeleton_by_dataset[dataset] = []
        skeleton_by_dataset[dataset].append(item['j2j_distance'])
    
    for item in skin_diffs:
        dataset = item['cls']
        if dataset not in skin_by_dataset:
            skin_by_dataset[dataset] = []
        skin_by_dataset[dataset].append(item['js_divergence'])
    
    print("Skeleton预测按数据集分析:")
    for dataset, distances in skeleton_by_dataset.items():
        mean_dist = np.mean(distances)
        max_dist = np.max(distances)
        print(f"  {dataset}: 平均J2J距离={mean_dist:.6f}, 最大={max_dist:.6f}, 样本数={len(distances)}")
    
    print("Skin预测按数据集分析:")
    for dataset, divergences in skin_by_dataset.items():
        mean_div = np.mean(divergences)
        max_div = np.max(divergences)
        print(f"  {dataset}: 平均JS散度={mean_div:.6f}, 最大={max_div:.6f}, 样本数={len(divergences)}")
    
    # 添加模型稳定性评估
    print("\n=== 模型稳定性评估 ===")
    
    # 计算变异系数 (CV = std/mean)
    if skeleton_diffs:
        skeleton_distances = [item['j2j_distance'] for item in skeleton_diffs]
        skeleton_cv = np.std(skeleton_distances) / np.mean(skeleton_distances)
        print(f"Skeleton预测变异系数: {skeleton_cv:.4f}")
        
        # 稳定性等级
        if skeleton_cv < 0.1:
            stability_level = "极高"
        elif skeleton_cv < 0.3:
            stability_level = "高"
        elif skeleton_cv < 0.5:
            stability_level = "中等"
        else:
            stability_level = "低"
        print(f"Skeleton预测稳定性等级: {stability_level}")
    
    if skin_diffs:
        skin_divergences = [item['js_divergence'] for item in skin_diffs]
        skin_cv = np.std(skin_divergences) / np.mean(skin_divergences)
        print(f"Skin预测变异系数: {skin_cv:.4f}")
        
        # 稳定性等级
        if skin_cv < 0.5:
            stability_level = "极高"
        elif skin_cv < 1.0:
            stability_level = "高"
        elif skin_cv < 2.0:
            stability_level = "中等"
        else:
            stability_level = "低"
        print(f"Skin预测稳定性等级: {stability_level}")
    
    # 保存详细的异常样本信息
    outlier_info = {
        'skeleton_outliers': skeleton_outliers if 'skeleton_outliers' in locals() else [],
        'skin_outliers': skin_outliers if 'skin_outliers' in locals() else [],
        'analysis_summary': {
            'total_samples': len(skeleton_diffs),
            'skeleton_stability': stability_level if skeleton_diffs else "N/A",
            'skin_stability': stability_level if skin_diffs else "N/A",
            'datasets_analyzed': list(skeleton_by_dataset.keys()) if skeleton_by_dataset else []
        }
    }
    
    with open(os.path.join(output_dir, "outlier_analysis.json"), 'w') as f:
        json.dump(outlier_info, f, indent=2)
    
    print(f"\n结果已保存到: {output_dir}")
    print(f"详细异常分析已保存到: {os.path.join(output_dir, 'outlier_analysis.json')}")

def explain_js_divergence():
    """
    解释JS散度的含义和计算方法
    """
    explanation = """
    === JS散度（Jensen-Shannon Divergence）详解 ===
    
    1. 定义：
       JS散度是一种衡量两个概率分布相似性的对称距离度量。
       它基于KL散度（Kullback-Leibler Divergence）构建。
    
    2. 数学公式：
       JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
       其中 M = 0.5 * (P + Q) 是两个分布的平均
       KL(P||Q) = Σ P(i) * log(P(i)/Q(i)) 是KL散度
    
    3. 特性：
       - 对称性：JS(P||Q) = JS(Q||P)
       - 非负性：JS(P||Q) ≥ 0
       - 有界性：0 ≤ JS(P||Q) ≤ log(2) ≈ 0.693
       - 当P=Q时，JS(P||Q) = 0
    
    4. 在皮肤权重预测中的意义：
       - 衡量两次预测的权重分布差异
       - 值越小表示两次预测的权重分布越相似
       - 0表示完全相同，0.693表示完全不同
    
    5. 实际应用解释：
       - JS散度 < 0.01: 两次预测几乎完全一致
       - JS散度 0.01-0.05: 两次预测高度相似，差异很小
       - JS散度 0.05-0.1: 两次预测相似，存在一定差异
       - JS散度 > 0.1: 两次预测存在显著差异
    
    6. 为什么使用JS散度而不是简单的L1/L2距离：
       - 考虑了权重的概率分布特性
       - 对权重归一化敏感
       - 更好地反映权重分配的语义差异
    """
    print(explanation)

def visualize_js_divergence_example():
    """
    提供JS散度计算的具体例子
    """
    print("\n=== JS散度计算示例 ===")
    
    # 示例1：完全相同的分布
    p1 = np.array([0.5, 0.3, 0.2])
    q1 = np.array([0.5, 0.3, 0.2])
    m1 = 0.5 * (p1 + q1)
    kl1_p = np.sum(p1 * np.log((p1 + 1e-8) / (m1 + 1e-8)))
    kl1_q = np.sum(q1 * np.log((q1 + 1e-8) / (m1 + 1e-8)))
    js1 = 0.5 * (kl1_p + kl1_q)
    print(f"示例1 - 相同分布:")
    print(f"  P = {p1}, Q = {q1}")
    print(f"  JS散度 = {js1:.6f} (应该接近0)")
    
    # 示例2：轻微差异的分布
    p2 = np.array([0.5, 0.3, 0.2])
    q2 = np.array([0.45, 0.35, 0.2])
    m2 = 0.5 * (p2 + q2)
    kl2_p = np.sum(p2 * np.log((p2 + 1e-8) / (m2 + 1e-8)))
    kl2_q = np.sum(q2 * np.log((q2 + 1e-8) / (m2 + 1e-8)))
    js2 = 0.5 * (kl2_p + kl2_q)
    print(f"\n示例2 - 轻微差异:")
    print(f"  P = {p2}, Q = {q2}")
    print(f"  JS散度 = {js2:.6f} (轻微差异)")
    
    # 示例3：显著差异的分布
    p3 = np.array([0.8, 0.1, 0.1])
    q3 = np.array([0.1, 0.1, 0.8])
    m3 = 0.5 * (p3 + q3)
    kl3_p = np.sum(p3 * np.log((p3 + 1e-8) / (m3 + 1e-8)))
    kl3_q = np.sum(q3 * np.log((q3 + 1e-8) / (m3 + 1e-8)))
    js3 = 0.5 * (kl3_p + kl3_q)
    print(f"\n示例3 - 显著差异:")
    print(f"  P = {p3}, Q = {q3}")
    print(f"  JS散度 = {js3:.6f} (显著差异)")
    
    print(f"\n在你的结果中，JS散度均值为0.005254，这表明:")
    print(f"  - 两次预测的皮肤权重分布非常相似")
    print(f"  - 模型具有很高的稳定性和一致性")
    print(f"  - 差异主要来自于模型的随机性而非系统性偏差")

def main():
    parser = argparse.ArgumentParser(description='计算两次预测结果的差值')
    
    parser.add_argument('--pred_dir1', type=str, 
                        default='submit/predict',
                        # default='result',
                        # default='predict/sal_t768_pc16384_w512_h8_e12_all_cos_emb_whnormals_normals_aug_re_eval_new',
                        help='第一次预测结果目录')
    parser.add_argument('--pred_dir2', type=str, 
                        default='predict/sal_t768_pc16384_w512_h8_e12_all_cos_emb_whnormals_normals_aug_re_eval_new_',
                        help='第二次预测结果目录')
    parser.add_argument('--output_dir', type=str, 
                        default='difference_analysis',
                        help='差值结果输出目录')
    parser.add_argument('--explain_js', action='store_true',
                        help='显示JS散度的详细解释')
    
    args = parser.parse_args()
    
    if args.explain_js:
        explain_js_divergence()
        visualize_js_divergence_example()
        return
    
    if not os.path.exists(args.pred_dir1):
        print(f"错误: 目录 {args.pred_dir1} 不存在")
        return
    
    if not os.path.exists(args.pred_dir2):
        print(f"错误: 目录 {args.pred_dir2} 不存在")
        return
    
    calculate_difference(args.pred_dir1, args.pred_dir2, args.output_dir)

if __name__ == '__main__':
    main()
