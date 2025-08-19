#!/usr/bin/env python3
"""
示例脚本：演示如何使用mesh内部约束损失

这个脚本展示了如何使用新添加的mesh_interior_loss函数来约束预测的骨骼关节点位于mesh模型内部。
"""

import jittor as jt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置Jittor使用CUDA
jt.flags.use_cuda = 1

# 导入我们的损失函数
from models.metrics import (
    mesh_interior_loss_vectorized, 
    mesh_interior_loss_advanced,
    skeleton_mesh_consistency_loss
)

def generate_sphere_mesh(radius=0.5, num_points=1000):
    """生成球体mesh的点云"""
    # 使用球坐标生成均匀分布的点
    phi = np.random.uniform(0, 2*np.pi, num_points)
    costheta = np.random.uniform(-1, 1, num_points)
    u = np.random.uniform(0, 1, num_points)
    
    theta = np.arccos(costheta)
    r = radius * np.cbrt(u)  # 立方根确保体积内均匀分布
    
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    return np.stack([x, y, z], axis=-1)

def generate_human_like_skeleton():
    """生成类人体骨骼关节点"""
    # 简化的人体骨骼（符合22个关节的定义）
    joints = np.array([
        [0.0, 0.0, 0.0],     # 0: hips
        [0.0, 0.1, 0.0],     # 1: spine
        [0.0, 0.2, 0.0],     # 2: chest
        [0.0, 0.3, 0.0],     # 3: upper_chest
        [0.0, 0.4, 0.0],     # 4: neck
        [0.0, 0.45, 0.0],    # 5: head
        [-0.1, 0.3, 0.0],    # 6: l_shoulder
        [-0.2, 0.2, 0.0],    # 7: l_upper_arm
        [-0.3, 0.1, 0.0],    # 8: l_lower_arm
        [-0.35, 0.0, 0.0],   # 9: l_hand
        [0.1, 0.3, 0.0],     # 10: r_shoulder
        [0.2, 0.2, 0.0],     # 11: r_upper_arm
        [0.3, 0.1, 0.0],     # 12: r_lower_arm
        [0.35, 0.0, 0.0],    # 13: r_hand
        [-0.05, -0.1, 0.0],  # 14: l_upper_leg
        [-0.05, -0.3, 0.0],  # 15: l_lower_leg
        [-0.05, -0.4, 0.0],  # 16: l_foot
        [-0.05, -0.45, 0.05],# 17: l_toe_base
        [0.05, -0.1, 0.0],   # 18: r_upper_leg
        [0.05, -0.3, 0.0],   # 19: r_lower_leg
        [0.05, -0.4, 0.0],   # 20: r_foot
        [0.05, -0.45, 0.05], # 21: r_toe_base
    ])
    return joints

def test_mesh_interior_loss():
    """测试mesh内部约束损失"""
    print("=== 测试Mesh内部约束损失 ===")
    
    batch_size = 2
    num_joints = 22
    num_vertices = 2048
    
    # 生成测试数据
    mesh_vertices = []
    for _ in range(batch_size):
        sphere_mesh = generate_sphere_mesh(radius=0.6, num_points=num_vertices)
        mesh_vertices.append(sphere_mesh)
    mesh_vertices = np.stack(mesh_vertices, axis=0)
    mesh_vertices = jt.array(mesh_vertices).float32()
    
    # 生成骨骼关节点
    skeleton = generate_human_like_skeleton()
    
    # 测试场景1：关节点在内部（应该有较小的损失）
    interior_joints = np.tile(skeleton, (batch_size, 1, 1))
    interior_joints = jt.array(interior_joints).float32()
    
    # 测试场景2：关节点在外部（应该有较大的损失）
    exterior_joints = interior_joints * 1.5  # 放大1.5倍，使其超出球体
    
    # 计算损失
    interior_loss = mesh_interior_loss_vectorized(interior_joints, mesh_vertices)
    exterior_loss = mesh_interior_loss_vectorized(exterior_joints, mesh_vertices)
    
    print(f"内部关节点损失: {interior_loss.item():.6f}")
    print(f"外部关节点损失: {exterior_loss.item():.6f}")
    print(f"损失差异: {(exterior_loss - interior_loss).item():.6f}")
    
    # 验证外部损失应该大于内部损失
    assert exterior_loss > interior_loss, "外部关节点的损失应该大于内部关节点"
    
    print("✓ 测试通过：外部关节点确实产生了更大的损失")
    
    return interior_joints, exterior_joints, mesh_vertices

def test_comprehensive_loss():
    """测试综合的骨骼-mesh一致性损失"""
    print("\n=== 测试综合一致性损失 ===")
    
    batch_size = 2
    
    # 生成测试数据
    mesh_vertices = []
    for _ in range(batch_size):
        sphere_mesh = generate_sphere_mesh(radius=0.6, num_points=1024)
        mesh_vertices.append(sphere_mesh)
    mesh_vertices = np.stack(mesh_vertices, axis=0)
    mesh_vertices = jt.array(mesh_vertices).float32()
    
    # 生成骨骼关节点
    skeleton = generate_human_like_skeleton()
    joints = np.tile(skeleton, (batch_size, 1, 1))
    joints = jt.array(joints).float32()
    
    # 计算综合损失
    total_loss, loss_dict = skeleton_mesh_consistency_loss(
        joints, mesh_vertices, 
        interior_weight=1.0, 
        smoothness_weight=0.1
    )
    
    print(f"总损失: {total_loss.item():.6f}")
    print(f"内部约束损失: {loss_dict['interior'].item():.6f}")
    print(f"平滑性损失: {loss_dict['smoothness'].item():.6f}")
    
    return total_loss, loss_dict

def visualize_results(interior_joints, exterior_joints, mesh_vertices):
    """可视化结果"""
    print("\n=== 生成可视化图像 ===")
    
    # 转换为numpy进行可视化
    interior_np = interior_joints[0].numpy()  # 取第一个batch
    exterior_np = exterior_joints[0].numpy()
    mesh_np = mesh_vertices[0].numpy()
    
    # 创建3D图
    fig = plt.figure(figsize=(15, 5))
    
    # 子图1：mesh点云
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(mesh_np[:, 0], mesh_np[:, 1], mesh_np[:, 2], 
                c='lightblue', alpha=0.3, s=1)
    ax1.set_title('Mesh Point Cloud')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # 子图2：内部关节点
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(mesh_np[:, 0], mesh_np[:, 1], mesh_np[:, 2], 
                c='lightblue', alpha=0.3, s=1)
    ax2.scatter(interior_np[:, 0], interior_np[:, 1], interior_np[:, 2], 
                c='green', s=50, label='Interior Joints')
    ax2.set_title('Interior Joints (Low Loss)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    
    # 子图3：外部关节点
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(mesh_np[:, 0], mesh_np[:, 1], mesh_np[:, 2], 
                c='lightblue', alpha=0.3, s=1)
    ax3.scatter(exterior_np[:, 0], exterior_np[:, 1], exterior_np[:, 2], 
                c='red', s=50, label='Exterior Joints')
    ax3.set_title('Exterior Joints (High Loss)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('/home/hxgk/MoGen/jittor-comp-human/mesh_interior_loss_demo.png', 
                dpi=300, bbox_inches='tight')
    print("✓ 可视化图像已保存到: mesh_interior_loss_demo.png")

def demonstrate_gradient_flow():
    """演示梯度反向传播"""
    print("\n=== 演示梯度反向传播 ===")
    
    batch_size = 1
    num_joints = 22
    
    # 生成mesh
    mesh_vertices = generate_sphere_mesh(radius=0.5, num_points=1024)
    mesh_vertices = jt.array(mesh_vertices).unsqueeze(0).float32()  # [1, N, 3]
    
    # 创建可学习的关节点参数
    skeleton = generate_human_like_skeleton()
    joints = jt.array(skeleton).unsqueeze(0).float32()  # [1, J, 3]
    joints.requires_grad = True
    
    # 将关节点移到外部
    joints_exterior = joints * 1.2
    joints_exterior.requires_grad = True
    
    # 计算损失
    loss = mesh_interior_loss_vectorized(joints_exterior, mesh_vertices)
    
    # 反向传播
    loss.backward()
    
    print(f"损失值: {loss.item():.6f}")
    print(f"梯度范数: {jt.norm(joints_exterior.grad).item():.6f}")
    print(f"平均梯度大小: {jt.mean(jt.abs(joints_exterior.grad)).item():.6f}")
    
    # 验证梯度存在
    assert joints_exterior.grad is not None, "梯度应该存在"
    assert jt.norm(joints_exterior.grad).item() > 0, "梯度应该非零"
    
    print("✓ 梯度反向传播正常工作")

def main():
    """主函数"""
    print("Mesh Interior Constraint Loss 测试和演示")
    print("=" * 50)
    
    # 设置随机种子以确保可重复性
    jt.set_global_seed(42)
    np.random.seed(42)
    
    try:
        # 运行测试
        interior_joints, exterior_joints, mesh_vertices = test_mesh_interior_loss()
        test_comprehensive_loss()
        demonstrate_gradient_flow()
        
        # 生成可视化
        visualize_results(interior_joints, exterior_joints, mesh_vertices)
        
        print("\n" + "=" * 50)
        print("✅ 所有测试通过！Mesh内部约束损失函数工作正常。")
        print("\n使用方法:")
        print("1. 在训练脚本中导入: from models.metrics import mesh_interior_loss_vectorized")
        print("2. 计算损失: interior_loss = mesh_interior_loss_vectorized(pred_joints, mesh_vertices)")
        print("3. 添加到总损失: total_loss += interior_weight * interior_loss")
        print("4. 调整参数: k_neighbors, margin 来控制约束强度")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise

if __name__ == "__main__":
    main()
