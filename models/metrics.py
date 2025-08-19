import jittor as jt
from dataset.format import symmetric_bones, symmetric_joint_pairs, parents

def J2J(
    joints_a: jt.Var,
    joints_b: jt.Var,
) -> jt.Var:
    '''
    calculate J2J loss in [-1, 1]^3 cube
    joints_a: (J1, 3) joint
    joints_b: (J2, 3) joint
    '''
    assert isinstance(joints_a, jt.Var)
    assert isinstance(joints_b, jt.Var)
    assert joints_a.ndim == 2, "joints_a should be shape (J1, 3)"
    assert joints_b.ndim == 2, "joints_b should be shape (J2, 3)"
    dis1 = ((joints_a.unsqueeze(0) - joints_b.unsqueeze(1))**2).sum(dim=-1).sqrt()
    loss1 = dis1.min(dim=-1)
    dis2 = ((joints_b.unsqueeze(0) - joints_a.unsqueeze(1))**2).sum(dim=-1).sqrt()
    loss2 = dis2.min(dim=-1)
    return (loss1.mean() + loss2.mean()) / 2 / 2

def bone_length_symmetry_loss(pred, symmetric_bones=symmetric_bones):
    """
    对称骨骼长度损失。
    Args:
        pred: [B, J, 3]，预测的关键点位置

    Returns:
        scalar loss
    """
    loss = 0.0
    for (l_start, l_end), (r_start, r_end) in symmetric_bones:
        # 取出左右两段骨骼的端点
        l_vec = pred[:, l_end] - pred[:, l_start]  # [B, 3]
        r_vec = pred[:, r_end] - pred[:, r_start]  # [B, 3]

        # 计算长度（L2范数）
        l_len = jt.norm(l_vec, dim=1)  # [B]
        r_len = jt.norm(r_vec, dim=1)  # [B]

        # 损失：平方误差
        loss += jt.mean((l_len - r_len) ** 2)
    return loss / len(symmetric_bones)


def reflect_across_plane(points, plane_point, plane_normal):
    """
    将点 points 绕平面 (plane_point, plane_normal) 做反射。
    Args:
        points: Jittor Var (B, N, 3)
        plane_point: Jittor Var (B, 3)
        plane_normal: Jittor Var (B, 3)
    Returns:
        reflected: Jittor Var (B, N, 3) 反射后的点
    """
    vec = points - plane_point.unsqueeze(1) # (B, N, 3) - (B, 1, 3)
    proj = (vec * plane_normal.unsqueeze(1)).sum(dim=-1, keepdims=True)  # (B, N, 1)
    reflected = points - 2 * proj * plane_normal.unsqueeze(1) # (B, N, 3)
    return reflected

def find_reflection_plane_colomna(target, pairs=None):
    """
    通过脊柱节点(0,1,2,3,4,5)计算人体的对称平面
    Args:
        target: Jittor Var (B, N, 3) Joints
    return: 
        c (B, 3), n (B, 3) 表示过点c、法向量为n的对称平面
    """
    # 获取脊柱节点
    spine_nodes = target[:, 0:6, :]  # (B, 6, 3)
    
    # 1. 计算脊柱节点的中心点作为平面中心
    c = spine_nodes.mean(dim=1)  # (B, 3)
    
    # 2. 计算脊柱方向向量
    spine_start = spine_nodes[:, 0, :]  # (B, 3)
    spine_end = spine_nodes[:, -1, :]   # (B, 3)
    spine_dir = spine_end - spine_start  # (B, 3)
    # 避免除以零
    spine_dir_norm = jt.norm(spine_dir, dim=-1, keepdims=True)
    spine_dir = spine_dir / (spine_dir_norm + 1e-6)
    
    # 3. 计算垂直于脊柱方向的向量作为平面法向量
    # 使用脊柱节点拟合一个平面
    spine_centered = spine_nodes - c.unsqueeze(1)  # (B, 6, 3)
    
    # 计算协方差矩阵 (Jittor的batch matmul)
    # (B, 3, 6) @ (B, 6, 3) -> (B, 3, 3)
    cov = jt.matmul(spine_centered.permute(0, 2, 1), spine_centered) 
    
    # 计算特征值和特征向量
    U, S, V = jt.linalg.svd(cov)
    
    # 取最小特征值对应的特征向量作为平面法向量
    n = U[:, :, -1]  # (B, 3)
    
    # 确保法向量方向一致性：如果法向量与脊柱方向点积为负，则反转法向量
    dot_product = (n * spine_dir).sum(dim=-1, keepdims=True)
    n = jt.where(dot_product < 0, -n, n) 
    
    # 单位化法向量
    n_norm = jt.norm(n, dim=-1, keepdims=True)
    n = n / (n_norm + 1e-6)
    
    return c, n

def symmetry_loss(pred, target, pairs=symmetric_joint_pairs):
    """
    每个样本独立推断对称平面，根据反射点计算对称损失
    pred: (B, J, 3)
    target: (B, J, 3)
    pairs: 对称点对列表
    """
    if pairs is None or len(pairs) == 0:
        return jt.zeros(1)

    plane_c, plane_n = find_reflection_plane_colomna(target, pairs)  # (B, 3), (B, 3)
    loss = jt.zeros(1) # Initialize loss as Jittor tensor

    for left_idx, right_idx in pairs:
        left_joint_pred = pred[:, left_idx, :]         # (B, 3)
        right_joint_pred = pred[:, right_idx, :]       # (B, 3)
        
        # 反射预测的右关节
        right_mirrored_pred = reflect_across_plane(right_joint_pred.unsqueeze(1), plane_c, plane_n).squeeze(1) # (B, 3)
        
        # 计算对称损失
        loss += jt.norm(left_joint_pred - right_mirrored_pred, dim=-1).mean()

    return loss / len(pairs)

def topology_loss(pred, target, parents=parents):
    """保持父子节点之间的相对向量结构一致"""
    loss = jt.zeros(1) # Initialize loss as Jittor tensor
    for i, p in enumerate(parents):
        if p is None: continue # Use 'is None' for None check
        pred_vec = pred[:, i, :] - pred[:, p, :]
        target_vec = target[:, i, :] - target[:, p, :]
        loss += jt.norm(pred_vec - target_vec, dim=-1).mean()
    return loss

def chamfer_distance(predict, target):
    """
    Args:
        predict: Jittor Var (B, N, 3)
        target: Jittor Var (B, M, 3)

    Return: 
        Jittor Var (scalar) Chamfer Distance
    """
    # 计算 predict 中每个点到 target 中所有点的平方欧氏距离
    dist_ab = jt.sum((predict.unsqueeze(2) - target.unsqueeze(1))**2, dim=-1) # (B, N, M)
    dist_a_to_b = dist_ab.min(dim=-1, keepdims=False) # (B, N)

    # 计算 target 中每个点到 predict 中所有点的平方欧氏距离
    dist_ba = jt.sum((target.unsqueeze(2) - predict.unsqueeze(1))**2, dim=-1) # (B, M, N)
    dist_b_to_a = dist_ba.min(dim=-1, keepdims=False) # (B, M)

    cd_sq_per_sample = dist_a_to_b.mean(dim=-1) + dist_b_to_a.mean(dim=-1) # (B,)
    return cd_sq_per_sample.mean()

def relative_position_loss(pred, target):
    """骨骼中心和相对位置一致性"""
    pcenter = pred.mean(dim=1, keepdims=True)
    tcenter = target.mean(dim=1, keepdims=True)
    prel = pred - pcenter
    trel = target - tcenter
    loss = jt.norm(prel - trel, dim=-1).mean()
    return loss

def mesh_interior_loss(pred_joints, mesh_vertices, k_neighbors=50, margin=0.01):
    """
    约束预测的骨骼关节点位于mesh模型内部
    
    Args:
        pred_joints: [B, J, 3] 预测的关节点
        mesh_vertices: [B, N, 3] mesh的顶点云
        k_neighbors: int, 用于估计局部密度的邻居数量
        margin: float, 内部边界的安全边距
        
    Returns:
        loss: 内部约束损失
    """
    batch_size, num_joints, _ = pred_joints.shape
    batch_size_mesh, num_vertices, _ = mesh_vertices.shape
    
    if batch_size != batch_size_mesh:
        raise ValueError(f"Batch size mismatch: joints {batch_size} vs mesh {batch_size_mesh}")
    
    total_loss = jt.zeros(1)
    
    for b in range(batch_size):
        joints_b = pred_joints[b]  # [J, 3]
        mesh_b = mesh_vertices[b]  # [N, 3]
        
        # 对每个关节点计算到mesh表面的距离
        for j in range(num_joints):
            joint = joints_b[j:j+1]  # [1, 3]
            
            # 计算关节点到所有mesh顶点的距离
            distances = jt.norm(mesh_b - joint, dim=-1)  # [N]
            
            # 找到k个最近的邻居
            k = min(k_neighbors, num_vertices)
            _, nearest_indices = jt.topk(-distances, k)  # 使用负距离来找最小值
            nearest_vertices = mesh_b[nearest_indices]  # [k, 3]
            nearest_distances = distances[nearest_indices]  # [k]
            
            # 计算局部密度权重（距离越近权重越大）
            weights = 1.0 / (nearest_distances + 1e-6)  # [k]
            weights = weights / weights.sum()  # 归一化
            
            # 计算加权平均距离作为到表面的近似距离
            surface_distance = (nearest_distances * weights).sum()
            
            # 如果关节点距离表面太近（可能在外部），施加惩罚
            # 使用smooth L1 loss来避免梯度爆炸
            if surface_distance < margin:
                penalty = margin - surface_distance
                total_loss += jt.where(penalty < 1.0, 
                                     0.5 * penalty * penalty, 
                                     penalty - 0.5)
    
    return total_loss / (batch_size * num_joints)

def mesh_interior_loss_vectorized(pred_joints, mesh_vertices, k_neighbors=50, margin=0.01):
    """
    向量化版本的mesh内部约束损失（更高效）
    
    Args:
        pred_joints: [B, J, 3] 预测的关节点
        mesh_vertices: [B, N, 3] mesh的顶点云
        k_neighbors: int, 用于估计局部密度的邻居数量
        margin: float, 内部边界的安全边距
        
    Returns:
        loss: 内部约束损失
    """
    batch_size, num_joints, _ = pred_joints.shape
    batch_size_mesh, num_vertices, _ = mesh_vertices.shape
    
    if batch_size != batch_size_mesh:
        raise ValueError(f"Batch size mismatch: joints {batch_size} vs mesh {batch_size_mesh}")
    
    # 计算所有关节点到所有mesh顶点的距离
    # pred_joints: [B, J, 3] -> [B, J, 1, 3]
    # mesh_vertices: [B, N, 3] -> [B, 1, N, 3]
    distances = jt.norm(pred_joints.unsqueeze(2) - mesh_vertices.unsqueeze(1), dim=-1)  # [B, J, N]
    
    # 找到每个关节点的k个最近邻居
    k = min(k_neighbors, num_vertices)
    nearest_distances, _ = jt.topk(-distances, k, dim=-1)  # [B, J, k] 使用负距离找最小值
    nearest_distances = -nearest_distances  # 转回正距离
    
    # 计算权重（距离越近权重越大）
    weights = 1.0 / (nearest_distances + 1e-6)  # [B, J, k]
    weights = weights / weights.sum(dim=-1, keepdims=True)  # 归一化
    
    # 计算加权平均距离作为到表面的近似距离
    surface_distances = (nearest_distances * weights).sum(dim=-1)  # [B, J]
    
    # 计算惩罚：如果距离表面太近则施加损失
    penalties = jt.maximum(margin - surface_distances, jt.zeros_like(surface_distances))
    
    # 使用smooth L1 loss
    smooth_penalties = jt.where(penalties < 1.0, 0.5 * penalties * penalties, penalties - 0.5)
    
    return smooth_penalties.mean()

def compute_face_normals(vertices, faces):
    """
    计算面片法向量
    
    Args:
        vertices: [B, N, 3] 顶点坐标
        faces: [B, F, 3] 面片索引
        
    Returns:
        face_normals: [B, F, 3] 面片法向量
    """
    # 获取面片的三个顶点
    v0 = jt.gather(vertices, 1, faces[:, :, 0].unsqueeze(-1).expand(-1, -1, 3))
    v1 = jt.gather(vertices, 1, faces[:, :, 1].unsqueeze(-1).expand(-1, -1, 3))
    v2 = jt.gather(vertices, 1, faces[:, :, 2].unsqueeze(-1).expand(-1, -1, 3))
    
    # 计算边向量
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    # 计算法向量 (叉积)
    normals = jt.cross(edge1, edge2, dim=-1)
    
    # 归一化
    normals = normals / (jt.norm(normals, dim=-1, keepdims=True) + 1e-8)
    
    return normals

def point_inside_mesh_check(points, mesh_vertices, mesh_faces, face_normals=None):
    """
    使用射线投射法判断点是否在mesh内部
    
    Args:
        points: [B, P, 3] 待检测的点
        mesh_vertices: [B, N, 3] mesh顶点
        mesh_faces: [B, F, 3] mesh面片索引
        face_normals: [B, F, 3] 面片法向量（可选）
        
    Returns:
        inside_mask: [B, P] 布尔张量，True表示点在内部
    """
    batch_size, num_points, _ = points.shape
    
    if face_normals is None:
        face_normals = compute_face_normals(mesh_vertices, mesh_faces)
    
    # 简化版本：使用重心坐标法估计内外部
    # 为每个点找到最近的面片，然后根据法向量判断内外部
    inside_scores = jt.zeros((batch_size, num_points))
    
    for b in range(batch_size):
        points_b = points[b]  # [P, 3]
        vertices_b = mesh_vertices[b]  # [N, 3]
        faces_b = mesh_faces[b]  # [F, 3]
        normals_b = face_normals[b]  # [F, 3]
        
        # 计算每个点到每个面片中心的距离
        face_centers = (vertices_b[faces_b[:, 0]] + 
                       vertices_b[faces_b[:, 1]] + 
                       vertices_b[faces_b[:, 2]]) / 3.0  # [F, 3]
        
        for p in range(num_points):
            point = points_b[p]  # [3]
            
            # 找到最近的面片
            distances_to_faces = jt.norm(face_centers - point.unsqueeze(0), dim=-1)
            closest_face_idx = jt.argmin(distances_to_faces)
            
            # 计算从点到最近面片中心的向量
            vec_to_point = point - face_centers[closest_face_idx]
            
            # 计算与法向量的点积（负值表示内部）
            dot_product = jt.sum(vec_to_point * normals_b[closest_face_idx])
            inside_scores[b, p] = -dot_product  # 内部为正值
    
    return inside_scores > 0

def mesh_interior_loss_advanced(pred_joints, mesh_vertices, mesh_faces=None, 
                               k_neighbors=50, margin=0.01, use_normal_check=True,
                               interior_penalty_scale=2.0):
    """
    高级版本的mesh内部约束损失，使用法向量信息判断内外部
    
    Args:
        pred_joints: [B, J, 3] 预测的关节点
        mesh_vertices: [B, N, 3] mesh的顶点云
        mesh_faces: [B, F, 3] mesh的面片索引（可选）
        k_neighbors: int, 用于估计局部密度的邻居数量
        margin: float, 内部边界的安全边距
        use_normal_check: bool, 是否使用法向量检查
        interior_penalty_scale: float, 外部惩罚的缩放因子
        
    Returns:
        loss: 内部约束损失
    """
    batch_size, num_joints, _ = pred_joints.shape
    
    # 使用向量化版本作为基础
    base_loss = mesh_interior_loss_vectorized(pred_joints, mesh_vertices, k_neighbors, margin)
    
    if not use_normal_check or mesh_faces is None:
        return base_loss
    
    try:
        # 计算面片法向量
        face_normals = compute_face_normals(mesh_vertices, mesh_faces)
        
        # 判断关节点是否在mesh内部
        inside_mask = point_inside_mesh_check(pred_joints, mesh_vertices, mesh_faces, face_normals)
        
        # 计算额外的外部惩罚
        outside_mask = ~inside_mask  # 外部点
        outside_penalty = outside_mask.float().mean() * interior_penalty_scale
        
        # 组合损失
        total_loss = base_loss + outside_penalty
        
        return total_loss
        
    except Exception as e:
        # 如果高级方法失败，回退到基础方法
        print(f"Advanced interior loss failed, falling back to basic method: {e}")
        return base_loss

def mesh_interior_loss_with_sdf(pred_joints, mesh_vertices, mesh_faces=None,
                               k_neighbors=50, margin=0.01, sdf_weight=1.0):
    """
    基于近似有向距离场的mesh内部约束损失
    
    Args:
        pred_joints: [B, J, 3] 预测的关节点
        mesh_vertices: [B, N, 3] mesh的顶点云
        mesh_faces: [B, F, 3] mesh的面片索引（可选）
        k_neighbors: int, 用于估计局部密度的邻居数量
        margin: float, 内部边界的安全边距
        sdf_weight: float, SDF损失的权重
        
    Returns:
        loss: 内部约束损失
    """
    batch_size, num_joints, _ = pred_joints.shape
    batch_size_mesh, num_vertices, _ = mesh_vertices.shape
    
    if batch_size != batch_size_mesh:
        raise ValueError(f"Batch size mismatch: joints {batch_size} vs mesh {batch_size_mesh}")
    
    # 计算到表面的近似有向距离
    total_loss = jt.zeros(1)
    
    for b in range(batch_size):
        joints_b = pred_joints[b]  # [J, 3]
        vertices_b = mesh_vertices[b]  # [N, 3]
        
        # 计算每个关节点的有向距离
        for j in range(num_joints):
            joint = joints_b[j]  # [3]
            
            # 计算到所有顶点的距离
            distances = jt.norm(vertices_b - joint.unsqueeze(0), dim=-1)  # [N]
            
            # 找到k个最近邻
            k = min(k_neighbors, num_vertices)
            nearest_distances, nearest_indices = jt.topk(-distances, k)
            nearest_distances = -nearest_distances
            nearest_vertices = vertices_b[nearest_indices]  # [k, 3]
            
            # 估算局部表面法向量
            # 使用最近邻顶点的重心作为局部表面点
            surface_point = nearest_vertices.mean(dim=0)  # [3]
            
            # 计算从关节点到表面的向量
            to_surface = surface_point - joint  # [3]
            
            # 使用PCA估算局部表面法向量
            centered_vertices = nearest_vertices - surface_point.unsqueeze(0)  # [k, 3]
            
            if k >= 3:
                # 计算协方差矩阵
                cov = jt.matmul(centered_vertices.transpose(0, 1), centered_vertices) / (k - 1)
                
                try:
                    # SVD分解获取主方向
                    U, S, V = jt.linalg.svd(cov)
                    # 最小特征值对应的特征向量作为法向量
                    normal = U[:, -1]
                    
                    # 计算有向距离
                    distance_to_surface = jt.norm(to_surface)
                    dot_product = jt.sum(to_surface * normal) / (distance_to_surface + 1e-8)
                    
                    # 如果点积为负，说明关节点在表面外侧
                    signed_distance = distance_to_surface * jt.sign(dot_product)
                    
                    # 外部点的惩罚（signed_distance > 0）
                    if signed_distance > -margin:
                        penalty = jt.maximum(signed_distance + margin, jt.zeros(1))
                        total_loss += sdf_weight * penalty * penalty
                        
                except:
                    # SVD失败时回退到简单距离
                    min_distance = nearest_distances.min()
                    if min_distance < margin:
                        penalty = margin - min_distance
                        total_loss += sdf_weight * penalty * penalty
            else:
                # 邻居太少时使用简单距离
                min_distance = nearest_distances.min()
                if min_distance < margin:
                    penalty = margin - min_distance
                    total_loss += sdf_weight * penalty * penalty
    
    return total_loss / (batch_size * num_joints)

def mesh_interior_loss_with_normals(pred_joints, mesh_vertices, mesh_normals, 
                                   k_neighbors=50, margin=0.01, normal_weight=1.0):
    """
    基于顶点法向量的mesh内部约束损失（更高效的实现）
    
    Args:
        pred_joints: [B, J, 3] 预测的关节点
        mesh_vertices: [B, N, 3] mesh的顶点云
        mesh_normals: [B, N, 3] mesh顶点的法向量
        k_neighbors: int, 用于估计局部密度的邻居数量
        margin: float, 内部边界的安全边距
        normal_weight: float, 法向量约束的权重
        
    Returns:
        loss: 内部约束损失
    """
    batch_size, num_joints, _ = pred_joints.shape
    batch_size_mesh, num_vertices, _ = mesh_vertices.shape
    
    if batch_size != batch_size_mesh:
        raise ValueError(f"Batch size mismatch: joints {batch_size} vs mesh {batch_size_mesh}")
    
    # 计算所有关节点到所有mesh顶点的距离
    # pred_joints: [B, J, 3] -> [B, J, 1, 3]
    # mesh_vertices: [B, N, 3] -> [B, 1, N, 3]
    distances = jt.norm(pred_joints.unsqueeze(2) - mesh_vertices.unsqueeze(1), dim=-1)  # [B, J, N]
    
    # 找到每个关节点的k个最近邻居
    k = min(k_neighbors, num_vertices)
    nearest_distances, nearest_indices = jt.topk(-distances, k, dim=-1)  # [B, J, k]
    nearest_distances = -nearest_distances  # 转回正距离
    
    # 获取最近邻顶点和对应的法向量
    batch_indices = jt.arange(batch_size).view(-1, 1, 1).expand(-1, num_joints, k)
    joint_indices = jt.arange(num_joints).view(1, -1, 1).expand(batch_size, -1, k)
    
    nearest_vertices = mesh_vertices[batch_indices, nearest_indices]  # [B, J, k, 3]
    nearest_normals = mesh_normals[batch_indices, nearest_indices]    # [B, J, k, 3]
    
    # 计算从最近邻顶点到关节点的向量
    joint_expanded = pred_joints.unsqueeze(2).expand(-1, -1, k, -1)  # [B, J, k, 3]
    to_joint_vectors = joint_expanded - nearest_vertices  # [B, J, k, 3]
    
    # 计算与法向量的点积
    # 如果点积为正，说明关节点在法向量指向的一侧（通常是外部）
    dot_products = (to_joint_vectors * nearest_normals).sum(dim=-1)  # [B, J, k]
    
    # 计算权重（距离越近权重越大）
    weights = 1.0 / (nearest_distances + 1e-6)  # [B, J, k]
    weights = weights / weights.sum(dim=-1, keepdims=True)  # 归一化
    
    # 加权平均点积值
    weighted_dot_products = (dot_products * weights).sum(dim=-1)  # [B, J]
    
    # 基础距离约束
    surface_distances = (nearest_distances * weights).sum(dim=-1)  # [B, J]
    distance_penalties = jt.maximum(margin - surface_distances, jt.zeros_like(surface_distances))
    
    # 法向量约束：如果加权点积为正，说明倾向于在外部
    normal_penalties = jt.maximum(weighted_dot_products, jt.zeros_like(weighted_dot_products))
    
    # 组合损失
    total_penalties = distance_penalties + normal_weight * normal_penalties
    
    # 使用smooth L1 loss
    smooth_penalties = jt.where(total_penalties < 1.0, 
                               0.5 * total_penalties * total_penalties, 
                               total_penalties - 0.5)
    
    return smooth_penalties.mean()

def mesh_interior_loss_fast_normals(pred_joints, mesh_vertices, mesh_normals, margin=0.05):
    """
    基于法向量的快速内部约束损失（最简化版本）
    
    Args:
        pred_joints: [B, J, 3] 预测的关节点
        mesh_vertices: [B, N, 3] mesh的顶点云
        mesh_normals: [B, N, 3] mesh顶点的法向量
        margin: float, 内部边界的安全边距
        
    Returns:
        loss: 内部约束损失
    """
    batch_size, num_joints, _ = pred_joints.shape
    
    # 对每个关节点，找到最近的mesh顶点
    distances = jt.norm(pred_joints.unsqueeze(2) - mesh_vertices.unsqueeze(1), dim=-1)  # [B, J, N]
    closest_indices = jt.argmin(distances, dim=-1)  # [B, J]
    
    # 获取最近顶点和对应法向量
    batch_indices = jt.arange(batch_size).view(-1, 1).expand(-1, num_joints)
    
    closest_vertices = mesh_vertices[batch_indices, closest_indices]  # [B, J, 3]
    closest_normals = mesh_normals[batch_indices, closest_indices]    # [B, J, 3]
    closest_distances = distances[batch_indices, jt.arange(num_joints).view(1, -1).expand(batch_size, -1), closest_indices]  # [B, J]
    
    # 计算从最近顶点到关节点的向量
    to_joint_vectors = pred_joints - closest_vertices  # [B, J, 3]
    
    # 计算与法向量的点积
    dot_products = (to_joint_vectors * closest_normals).sum(dim=-1)  # [B, J]
    
    # 惩罚函数：
    # 1. 距离太近的惩罚
    distance_penalty = jt.maximum(margin - closest_distances, jt.zeros_like(closest_distances))
    
    # 2. 在外部的惩罚（点积为正）
    outside_penalty = jt.maximum(dot_products / (jt.norm(to_joint_vectors, dim=-1) + 1e-8), 
                                jt.zeros_like(dot_products))
    
    # 组合惩罚
    total_penalty = distance_penalty + outside_penalty
    
    return total_penalty.mean()

def mesh_interior_loss_stable(pred_joints, mesh_vertices, mesh_normals, 
                             k_neighbors=32, margin=0.02, temperature=0.1):
    """
    稳定版本的内部约束损失，使用软注意力机制
    
    Args:
        pred_joints: [B, J, 3] 预测的关节点
        mesh_vertices: [B, N, 3] mesh的顶点云
        mesh_normals: [B, N, 3] mesh顶点的法向量
        k_neighbors: int, 近邻点数量
        margin: float, 内部边界的安全边距
        temperature: float, 软注意力的温度参数
        
    Returns:
        loss: 内部约束损失
    """
    batch_size, num_joints, _ = pred_joints.shape
    
    # 计算距离矩阵
    distances = jt.norm(pred_joints.unsqueeze(2) - mesh_vertices.unsqueeze(1), dim=-1)  # [B, J, N]
    
    # 找到k个最近邻
    k = min(k_neighbors, mesh_vertices.shape[1])
    nearest_distances, nearest_indices = jt.topk(-distances, k, dim=-1)  # [B, J, k]
    nearest_distances = -nearest_distances
    
    # 获取最近邻顶点和法向量
    batch_indices = jt.arange(batch_size).view(-1, 1, 1).expand(-1, num_joints, k)
    joint_indices = jt.arange(num_joints).view(1, -1, 1).expand(batch_size, -1, k)
    
    nearest_vertices = mesh_vertices.gather(dim=1, index=nearest_indices.view(batch_size, -1, 1).expand(-1, -1, 3)).view(batch_size, num_joints, k, 3)
    nearest_normals = mesh_normals.gather(dim=1, index=nearest_indices.view(batch_size, -1, 1).expand(-1, -1, 3)).view(batch_size, num_joints, k, 3)
    
    # 软注意力权重（基于距离的softmax）
    attention_weights = jt.nn.softmax(-nearest_distances / temperature, dim=-1)  # [B, J, k]
    
    # 加权平均顶点位置和法向量
    weighted_vertices = (nearest_vertices * attention_weights.unsqueeze(-1)).sum(dim=2)  # [B, J, 3]
    weighted_normals = (nearest_normals * attention_weights.unsqueeze(-1)).sum(dim=2)    # [B, J, 3]
    weighted_normals = weighted_normals / (jt.norm(weighted_normals, dim=-1, keepdims=True) + 1e-8)
    
    # 计算从加权表面点到关节点的向量
    to_joint_vectors = pred_joints - weighted_vertices  # [B, J, 3]
    
    # 计算有向距离（点积）
    signed_distances = (to_joint_vectors * weighted_normals).sum(dim=-1)  # [B, J]
    actual_distances = jt.norm(to_joint_vectors, dim=-1)  # [B, J]
    
    # 多重约束损失
    # 1. 距离约束：关节点不应距离表面太近
    distance_penalty = jt.maximum(margin - actual_distances, jt.zeros_like(actual_distances))
    
    # 2. 方向约束：关节点应在法向量指向内部的一侧
    direction_penalty = jt.maximum(signed_distances, jt.zeros_like(signed_distances))
    
    # 3. 平滑约束：使用平滑损失函数
    def smooth_l1_loss(x, beta=1.0):
        return jt.where(jt.abs(x) < beta, 0.5 * x * x / beta, jt.abs(x) - 0.5 * beta)
    
    total_penalty = smooth_l1_loss(distance_penalty) + smooth_l1_loss(direction_penalty)
    
    return total_penalty.mean()

def mesh_interior_loss_hierarchical(pred_joints, mesh_vertices, mesh_normals, 
                                   max_samples=2048, k_neighbors=32, margin=0.02):
    """
    分层采样的高效内部约束损失
    
    Args:
        pred_joints: [B, J, 3] 预测的关节点
        mesh_vertices: [B, N, 3] mesh的顶点云
        mesh_normals: [B, N, 3] mesh顶点的法向量
        max_samples: int, 最大采样点数
        k_neighbors: int, 近邻点数量
        margin: float, 内部边界的安全边距
        
    Returns:
        loss: 内部约束损失
    """
    batch_size, num_joints, _ = pred_joints.shape
    num_vertices = mesh_vertices.shape[1]
    
    # 如果顶点数量太多，进行分层采样
    if num_vertices > max_samples:
        # 使用FPS进行采样
        sampled_vertices, sampled_normals = hierarchical_sampling(
            mesh_vertices, mesh_normals, max_samples, pred_joints
        )
    else:
        sampled_vertices = mesh_vertices
        sampled_normals = mesh_normals
    
    # 使用采样后的点进行稳定的内部约束计算
    return mesh_interior_loss_stable(pred_joints, sampled_vertices, sampled_normals, 
                                   k_neighbors, margin)

def hierarchical_sampling(vertices, normals, max_samples, reference_points):
    """
    基于参考点的分层采样
    
    Args:
        vertices: [B, N, 3] 原始顶点
        normals: [B, N, 3] 原始法向量
        max_samples: int, 目标采样数量
        reference_points: [B, J, 3] 参考点（关节点）
        
    Returns:
        sampled_vertices: [B, max_samples, 3]
        sampled_normals: [B, max_samples, 3]
    """
    batch_size, num_vertices, _ = vertices.shape
    
    if num_vertices <= max_samples:
        return vertices, normals
    
    sampled_vertices_list = []
    sampled_normals_list = []
    
    for b in range(batch_size):
        verts_b = vertices[b]  # [N, 3]
        norms_b = normals[b]   # [N, 3]
        refs_b = reference_points[b]  # [J, 3]
        
        # 计算每个顶点到所有参考点的最小距离
        distances_to_refs = jt.norm(verts_b.unsqueeze(1) - refs_b.unsqueeze(0), dim=-1)  # [N, J]
        min_distances = distances_to_refs.min(dim=-1)[0]  # [N]
        
        # 基于距离的重要性采样
        # 距离关节点近的顶点更重要
        importance_weights = 1.0 / (min_distances + 1e-3)
        importance_weights = importance_weights / importance_weights.sum()
        
        # 使用重要性权重进行采样
        sampled_indices = jt.multinomial(importance_weights, max_samples, replacement=True)
        
        sampled_verts = verts_b[sampled_indices]
        sampled_norms = norms_b[sampled_indices]
        
        sampled_vertices_list.append(sampled_verts)
        sampled_normals_list.append(sampled_norms)
    
    return jt.stack(sampled_vertices_list), jt.stack(sampled_normals_list)

def skeleton_mesh_consistency_loss(pred_joints, mesh_vertices, mesh_normals=None, mesh_faces=None, parents=parents, 
                                 interior_weight=1.0, smoothness_weight=0.1, use_advanced_interior=False, use_normals=True):
    """
    综合的骨骼-mesh一致性损失
    
    Args:
        pred_joints: [B, J, 3] 预测的关节点
        mesh_vertices: [B, N, 3] mesh的顶点云
        mesh_normals: [B, N, 3] mesh顶点的法向量（可选）
        mesh_faces: [B, F, 3] mesh的面片索引（可选）
        parents: list, 父节点索引
        interior_weight: float, 内部约束权重
        smoothness_weight: float, 平滑性约束权重
        use_advanced_interior: bool, 是否使用高级内部约束
        use_normals: bool, 是否使用法向量进行内部约束
        
    Returns:
        total_loss: 总一致性损失
        loss_dict: 各项损失的详细信息
    """
    # 1. 内部约束损失 - 优先使用法向量方法
    if use_normals and mesh_normals is not None:
        # 使用基于法向量的快速方法
        interior_loss = mesh_interior_loss_fast_normals(pred_joints, mesh_vertices, mesh_normals)
    elif use_advanced_interior and mesh_faces is not None:
        # 使用高级面片方法
        interior_loss = mesh_interior_loss_advanced(pred_joints, mesh_vertices, mesh_faces)
    else:
        # 回退到基础向量化方法
        interior_loss = mesh_interior_loss_vectorized(pred_joints, mesh_vertices)
    
    # 2. 骨骼平滑性约束（相邻关节点不应该有突变）
    smoothness_loss = jt.zeros(1)
    valid_bones = 0
    
    for i, parent_idx in enumerate(parents):
        if parent_idx is not None:
            # 计算父子关节之间的骨骼向量
            bone_vectors = pred_joints[:, i] - pred_joints[:, parent_idx]  # [B, 3]
            
            # 计算骨骼长度变化的平滑性
            bone_lengths = jt.norm(bone_vectors, dim=-1)  # [B]
            
            # 批次内的长度变化应该相对平滑
            if bone_lengths.shape[0] > 1:
                length_variance = jt.var(bone_lengths)
                smoothness_loss += length_variance
                valid_bones += 1
    
    if valid_bones > 0:
        smoothness_loss = smoothness_loss / valid_bones
    
    # 3. 骨骼方向一致性约束
    direction_loss = jt.zeros(1)
    direction_count = 0
    
    for i, parent_idx in enumerate(parents):
        if parent_idx is not None:
            # 计算父子关节之间的方向向量
            bone_vectors = pred_joints[:, i] - pred_joints[:, parent_idx]  # [B, 3]
            bone_directions = bone_vectors / (jt.norm(bone_vectors, dim=-1, keepdims=True) + 1e-8)
            
            # 批次内方向向量的一致性
            if bone_directions.shape[0] > 1:
                # 计算方向向量的方差
                direction_variance = jt.var(bone_directions, dim=0).sum()
                direction_loss += direction_variance
                direction_count += 1
    
    if direction_count > 0:
        direction_loss = direction_loss / direction_count
    
    # 4. 组合损失
    total_loss = (interior_weight * interior_loss + 
                 smoothness_weight * smoothness_loss + 
                 0.05 * direction_loss)  # 方向一致性权重较小
    
    loss_dict = {
        'interior': interior_loss,
        'smoothness': smoothness_loss,
        'direction': direction_loss,
        'total': total_loss
    }
    
    return total_loss, loss_dict

