#!/usr/bin/env python3
"""
测试优化后的内部约束损失实现
"""

import jittor as jt
import numpy as np
import time
from models.metrics import (
    mesh_interior_loss_vectorized, 
    mesh_interior_loss_advanced,
    mesh_interior_loss_with_sdf,
    skeleton_mesh_consistency_loss,
    compute_face_normals,
    point_inside_mesh_check
)

# 设置随机种子
jt.set_global_seed(42)
np.random.seed(42)

def create_test_data():
    """创建测试数据"""
    batch_size = 2
    num_joints = 24
    num_vertices = 1024
    num_faces = 512
    
    # 创建一个立方体形状的mesh
    vertices = jt.randn(batch_size, num_vertices, 3) * 0.8  # 稍小于立方体
    
    # 创建一些关节点，部分在内部，部分在外部
    joints_inside = jt.randn(batch_size, num_joints//2, 3) * 0.5  # 内部关节
    joints_outside = jt.randn(batch_size, num_joints//2, 3) * 1.5  # 外部关节
    joints = jt.concat([joints_inside, joints_outside], dim=1)
    
    # 创建简单的面片（三角形）
    faces = jt.randint(0, num_vertices, (batch_size, num_faces, 3))
    
    return joints, vertices, faces

def test_face_normal_computation():
    """测试面片法向量计算"""
    print("=== 测试面片法向量计算 ===")
    
    # 创建简单的三角形
    batch_size = 1
    vertices = jt.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]]).float32()
    faces = jt.array([[[0, 1, 2]]]).long()  # 一个三角形面片
    
    normals = compute_face_normals(vertices, faces)
    print(f"面片法向量形状: {normals.shape}")
    print(f"面片法向量: {normals}")
    
    # 验证法向量是否为单位向量
    norm_length = jt.norm(normals, dim=-1)
    print(f"法向量长度: {norm_length}")
    print()

def test_interior_point_detection():
    """测试内外部点检测"""
    print("=== 测试内外部点检测 ===")
    
    # 创建立方体顶点
    vertices = jt.array([[
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # 底面
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # 顶面
    ]]).float32()
    
    # 立方体的12个三角形面片
    faces = jt.array([[
        [0, 1, 2], [0, 2, 3],  # 底面
        [4, 7, 6], [4, 6, 5],  # 顶面
        [0, 4, 5], [0, 5, 1],  # 前面
        [2, 6, 7], [2, 7, 3],  # 后面
        [0, 3, 7], [0, 7, 4],  # 左面
        [1, 5, 6], [1, 6, 2]   # 右面
    ]]).long()
    
    # 测试点：内部和外部
    test_points = jt.array([[
        [0, 0, 0],      # 中心点 (内部)
        [0.5, 0.5, 0.5], # 内部点
        [2, 0, 0],      # 外部点
        [-2, -2, -2]    # 外部点
    ]]).float32()
    
    inside_mask = point_inside_mesh_check(test_points, vertices, faces)
    print(f"测试点: {test_points.squeeze()}")
    print(f"内部检测结果: {inside_mask}")
    print()

def test_loss_functions():
    """测试各种损失函数"""
    print("=== 测试损失函数 ===")
    
    joints, vertices, faces = create_test_data()
    
    # 测试基础向量化损失
    start_time = time.time()
    loss_basic = mesh_interior_loss_vectorized(joints, vertices)
    time_basic = time.time() - start_time
    print(f"基础向量化损失: {loss_basic.item():.6f}, 时间: {time_basic:.4f}s")
    
    # 测试高级损失
    start_time = time.time()
    loss_advanced = mesh_interior_loss_advanced(joints, vertices, faces)
    time_advanced = time.time() - start_time
    print(f"高级损失: {loss_advanced.item():.6f}, 时间: {time_advanced:.4f}s")
    
    # 测试SDF损失
    start_time = time.time()
    loss_sdf = mesh_interior_loss_with_sdf(joints, vertices, faces)
    time_sdf = time.time() - start_time
    print(f"SDF损失: {loss_sdf.item():.6f}, 时间: {time_sdf:.4f}s")
    
    # 测试综合一致性损失
    start_time = time.time()
    loss_consistency, loss_dict = skeleton_mesh_consistency_loss(
        joints, vertices, faces, use_advanced_interior=True
    )
    time_consistency = time.time() - start_time
    print(f"综合一致性损失: {loss_consistency.item():.6f}, 时间: {time_consistency:.4f}s")
    print(f"损失组成: {{k: v.item() for k, v in loss_dict.items() if k != 'total'}}")
    print()

def test_gradient_flow():
    """测试梯度流"""
    print("=== 测试梯度流 ===")
    
    joints, vertices, faces = create_test_data()
    joints.requires_grad = True
    
    # 计算损失
    loss = mesh_interior_loss_advanced(joints, vertices, faces)
    
    # 反向传播
    grad = jt.grad(loss, joints)
    
    print(f"损失值: {loss.item():.6f}")
    print(f"梯度形状: {grad.shape}")
    print(f"梯度范围: [{grad.min().item():.6f}, {grad.max().item():.6f}]")
    print(f"梯度均值: {grad.mean().item():.6f}")
    print()

def benchmark_performance():
    """性能基准测试"""
    print("=== 性能基准测试 ===")
    
    # 不同规模的测试
    test_configs = [
        (1, 24, 512),   # 小规模
        (4, 24, 1024),  # 中等规模
        (8, 24, 2048),  # 大规模
    ]
    
    for batch_size, num_joints, num_vertices in test_configs:
        print(f"测试配置: B={batch_size}, J={num_joints}, N={num_vertices}")
        
        # 创建测试数据
        joints = jt.randn(batch_size, num_joints, 3)
        vertices = jt.randn(batch_size, num_vertices, 3)
        faces = jt.randint(0, num_vertices, (batch_size, num_vertices//4, 3))
        
        # 预热
        for _ in range(3):
            _ = mesh_interior_loss_vectorized(joints, vertices)
        
        # 基础方法基准
        start_time = time.time()
        for _ in range(10):
            loss_basic = mesh_interior_loss_vectorized(joints, vertices)
        time_basic = (time.time() - start_time) / 10
        
        # 高级方法基准
        start_time = time.time()
        for _ in range(10):
            loss_advanced = mesh_interior_loss_advanced(joints, vertices, faces)
        time_advanced = (time.time() - start_time) / 10
        
        print(f"  基础方法: {time_basic:.4f}s")
        print(f"  高级方法: {time_advanced:.4f}s")
        print(f"  加速比: {time_basic/time_advanced:.2f}x")
        print()

def main():
    """主函数"""
    print("开始测试优化后的内部约束损失实现...")
    print("=" * 50)
    
    # 设置Jittor
    jt.flags.use_cuda = 1 if jt.has_cuda else 0
    print(f"使用设备: {'CUDA' if jt.flags.use_cuda else 'CPU'}")
    print()
    
    try:
        # 运行各项测试
        test_face_normal_computation()
        test_interior_point_detection()
        test_loss_functions()
        test_gradient_flow()
        benchmark_performance()
        
        print("=" * 50)
        print("所有测试完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
