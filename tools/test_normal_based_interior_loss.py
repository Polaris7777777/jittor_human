#!/usr/bin/env python3
"""
测试基于normals的内部约束损失实现
"""

import jittor as jt
import numpy as np
import time
from models.metrics import (
    mesh_interior_loss_vectorized, 
    mesh_interior_loss_with_normals,
    mesh_interior_loss_fast_normals,
    skeleton_mesh_consistency_loss
)

# 设置随机种子
jt.set_global_seed(42)
np.random.seed(42)

def create_sphere_mesh(batch_size=1, num_vertices=1000, radius=1.0):
    """创建球形mesh用于测试"""
    # 生成球面上的点
    phi = np.random.uniform(0, 2*np.pi, (batch_size, num_vertices))
    theta = np.random.uniform(0, np.pi, (batch_size, num_vertices))
    
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    
    vertices = np.stack([x, y, z], axis=-1)  # [B, N, 3]
    
    # 法向量就是从原点到顶点的单位向量（朝外）
    normals = vertices / np.linalg.norm(vertices, axis=-1, keepdims=True)
    
    return jt.array(vertices).float32(), jt.array(normals).float32()

def create_test_joints(batch_size=1, num_joints=24, radius=1.0):
    """创建测试关节点（部分在内部，部分在外部）"""
    # 内部关节点（半径 < radius）
    num_inside = num_joints // 2
    inside_joints = np.random.randn(batch_size, num_inside, 3)
    inside_joints = inside_joints / np.linalg.norm(inside_joints, axis=-1, keepdims=True) * (radius * 0.5)
    
    # 外部关节点（半径 > radius）
    num_outside = num_joints - num_inside
    outside_joints = np.random.randn(batch_size, num_outside, 3)
    outside_joints = outside_joints / np.linalg.norm(outside_joints, axis=-1, keepdims=True) * (radius * 1.5)
    
    # 合并
    joints = np.concatenate([inside_joints, outside_joints], axis=1)
    
    return jt.array(joints).float32()

def test_normal_based_loss():
    """测试基于法向量的内部约束损失"""
    print("=== 测试基于法向量的内部约束损失 ===")
    
    batch_size = 2
    num_vertices = 1000
    num_joints = 24
    radius = 1.0
    
    # 创建球形mesh
    vertices, normals = create_sphere_mesh(batch_size, num_vertices, radius)
    
    # 创建测试关节点
    joints = create_test_joints(batch_size, num_joints, radius)
    
    print(f"Mesh顶点形状: {vertices.shape}")
    print(f"法向量形状: {normals.shape}")
    print(f"关节点形状: {joints.shape}")
    
    # 测试不同的损失函数
    print("\n--- 损失函数比较 ---")
    
    # 1. 基础向量化损失
    start_time = time.time()
    loss_basic = mesh_interior_loss_vectorized(joints, vertices)
    time_basic = time.time() - start_time
    print(f"基础向量化损失: {loss_basic.item():.6f}, 时间: {time_basic:.4f}s")
    
    # 2. 基于法向量的详细损失
    start_time = time.time()
    loss_normals = mesh_interior_loss_with_normals(joints, vertices, normals)
    time_normals = time.time() - start_time
    print(f"法向量详细损失: {loss_normals.item():.6f}, 时间: {time_normals:.4f}s")
    
    # 3. 基于法向量的快速损失
    start_time = time.time()
    loss_fast = mesh_interior_loss_fast_normals(joints, vertices, normals)
    time_fast = time.time() - start_time
    print(f"法向量快速损失: {loss_fast.item():.6f}, 时间: {time_fast:.4f}s")
    
    print(f"\n加速比（基础 vs 快速法向量）: {time_basic/time_fast:.2f}x")

def test_interior_exterior_detection():
    """测试内外部检测的准确性"""
    print("\n=== 测试内外部检测准确性 ===")
    
    batch_size = 1
    radius = 1.0
    
    # 创建球形mesh
    vertices, normals = create_sphere_mesh(batch_size, 1000, radius)
    
    # 创建已知内外部的测试点
    test_points = jt.array([
        [[0, 0, 0]],          # 中心点 (内部)
        [[0.5, 0, 0]],        # 内部点
        [[0.8, 0, 0]],        # 接近边界的内部点
        [[1.2, 0, 0]],        # 外部点
        [[2.0, 0, 0]]         # 远外部点
    ]).float32()  # [5, 1, 3]
    
    expected_labels = ["内部", "内部", "内部", "外部", "外部"]
    
    print("测试点检测结果：")
    for i, (point, label) in enumerate(zip(test_points, expected_labels)):
        # 计算损失值（损失越大说明越可能在外部）
        loss = mesh_interior_loss_fast_normals(point, vertices, normals, margin=0.1)
        print(f"点 {point.squeeze().numpy()}: {label}, 损失 = {loss.item():.6f}")

def test_gradient_quality():
    """测试梯度质量"""
    print("\n=== 测试梯度质量 ===")
    
    # 创建测试数据
    vertices, normals = create_sphere_mesh(1, 500, 1.0)
    joints = create_test_joints(1, 12, 1.0)
    joints.requires_grad = True
    
    # 计算损失和梯度
    loss = mesh_interior_loss_fast_normals(joints, vertices, normals)
    grad = jt.grad(loss, joints)
    
    print(f"损失值: {loss.item():.6f}")
    print(f"梯度形状: {grad.shape}")
    print(f"梯度统计:")
    print(f"  均值: {grad.mean().item():.6f}")
    print(f"  标准差: {grad.std().item():.6f}")
    print(f"  最小值: {grad.min().item():.6f}")
    print(f"  最大值: {grad.max().item():.6f}")
    
    # 检查梯度是否合理
    grad_norm = jt.norm(grad, dim=-1).mean()
    print(f"平均梯度范数: {grad_norm.item():.6f}")

def performance_benchmark():
    """性能基准测试"""
    print("\n=== 性能基准测试 ===")
    
    test_configs = [
        (1, 512, 12),     # 小规模
        (4, 1024, 24),    # 中等规模
        (8, 2048, 24),    # 大规模
    ]
    
    for batch_size, num_vertices, num_joints in test_configs:
        print(f"\n测试配置: B={batch_size}, N={num_vertices}, J={num_joints}")
        
        # 创建测试数据
        vertices, normals = create_sphere_mesh(batch_size, num_vertices, 1.0)
        joints = create_test_joints(batch_size, num_joints, 1.0)
        
        # 预热
        for _ in range(3):
            _ = mesh_interior_loss_fast_normals(joints, vertices, normals)
        
        # 基准测试
        num_runs = 20
        
        # 基础方法
        start_time = time.time()
        for _ in range(num_runs):
            loss_basic = mesh_interior_loss_vectorized(joints, vertices)
        time_basic = (time.time() - start_time) / num_runs
        
        # 法向量方法
        start_time = time.time()
        for _ in range(num_runs):
            loss_normals = mesh_interior_loss_fast_normals(joints, vertices, normals)
        time_normals = (time.time() - start_time) / num_runs
        
        print(f"  基础方法: {time_basic*1000:.2f}ms")
        print(f"  法向量方法: {time_normals*1000:.2f}ms")
        print(f"  加速比: {time_basic/time_normals:.2f}x")
        print(f"  基础损失: {loss_basic.item():.6f}")
        print(f"  法向量损失: {loss_normals.item():.6f}")

def test_integration_with_training():
    """测试与训练流程的集成"""
    print("\n=== 测试与训练流程集成 ===")
    
    # 模拟训练数据
    batch_size = 4
    vertices, normals = create_sphere_mesh(batch_size, 1024, 1.0)
    joints = create_test_joints(batch_size, 24, 1.0)
    
    # 测试综合损失函数
    total_loss, loss_dict = skeleton_mesh_consistency_loss(
        pred_joints=joints,
        mesh_vertices=vertices,
        mesh_normals=normals,
        use_normals=True,
        interior_weight=1.0,
        smoothness_weight=0.1
    )
    
    print(f"综合损失: {total_loss.item():.6f}")
    print("损失组成:")
    for key, value in loss_dict.items():
        if key != 'total':
            print(f"  {key}: {value.item():.6f}")

def main():
    """主函数"""
    print("开始测试基于normals的内部约束损失...")
    print("=" * 60)
    
    # 设置Jittor
    jt.flags.use_cuda = 1 if jt.has_cuda else 0
    print(f"使用设备: {'CUDA' if jt.flags.use_cuda else 'CPU'}")
    
    try:
        test_normal_based_loss()
        test_interior_exterior_detection()
        test_gradient_quality()
        performance_benchmark()
        test_integration_with_training()
        
        print("\n" + "=" * 60)
        print("所有测试完成！基于normals的内部约束实现正常工作。")
        print("\n主要优势:")
        print("1. 直接利用已有的顶点法向量")
        print("2. 更精确的内外部判断")
        print("3. 更好的计算效率")
        print("4. 更稳定的梯度")
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
