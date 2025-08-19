import jittor as jt
import numpy as np
import os
from jittor.dataset import Dataset

import os
from typing import List, Dict, Callable, Union, Tuple

from .asset import Asset
from .sampler import Sampler
from .format import body_mask

from scipy.spatial.transform import Rotation as R
from .asset import axis_angle_to_matrix
import copy
import random

def transform(asset: Asset):
    """
    Transform the asset data into [-1, 1]^3.
    """
    # Find min and max values for each dimension of points
    min_vals = np.min(asset.vertices, axis=0)
    max_vals = np.max(asset.vertices, axis=0)
    
    # Calculate the center of the bounding box
    center = (min_vals + max_vals) / 2
    
    # Calculate the scale factor to normalize to [-1, 1]
    # We take the maximum range across all dimensions to preserve aspect ratio
    scale = np.max(max_vals - min_vals) / 2
    
    # Normalize points to [-1, 1]^3
    normalized_vertices = (asset.vertices - center) / scale
    
    # Apply the same transformation to joints
    if asset.joints is not None:
        normalized_joints = (asset.joints - center) / scale
    else:
        normalized_joints = None
    
    asset.vertices  = normalized_vertices
    asset.joints    = normalized_joints
    # remember to change matrix_local !
    asset.matrix_local[:, :3, 3] = normalized_joints

def aug_transform(asset: Asset, 
              enable_rotation: bool = False,
              enable_scaling: bool = False,
              rotation_range: float = 180.0,
              scaling_range: Tuple[float, float] = (0.8, 1.2),
              normalize: bool = False):
    """
    Transform the asset data with data augmentation including random rotation and scaling.
    
    Args:
        asset: Asset object to transform
        enable_rotation: Whether to apply random rotation
        enable_scaling: Whether to apply random scaling  
        rotation_range: Maximum rotation angle in degrees (applied to each axis)
        scaling_range: (min_scale, max_scale) for uniform scaling
        normalize: Whether to normalize to [-1, 1]^3
    """
    vertices = asset.vertices.copy()
    joints = asset.joints.copy() if asset.joints is not None else None
    
    # 1. Random Rotation (if enabled)
    if enable_rotation:
        # Generate random rotation angles for each axis
        rotation_angles = (np.random.rand(3) - 0.5) * 2 * rotation_range
        rotation_matrix = R.from_euler('xyz', rotation_angles, degrees=True).as_matrix()
        
        # Apply rotation to vertices
        vertices = vertices @ rotation_matrix.T
        
        # Apply rotation to joints
        if joints is not None:
            joints = joints @ rotation_matrix.T
            
        # Update normals if they exist
        if asset.vertex_normals is not None:
            asset.vertex_normals = asset.vertex_normals @ rotation_matrix.T
        if asset.face_normals is not None:
            asset.face_normals = asset.face_normals @ rotation_matrix.T
    
    # 2. Random Scaling (if enabled)
    if enable_scaling:
        # Generate random uniform scaling factor
        scale_factor = np.random.uniform(scaling_range[0], scaling_range[1])
        
        # Apply scaling
        vertices = vertices * scale_factor
        if joints is not None:
            joints = joints * scale_factor
    
    # 3. Normalization to [-1, 1]^3 (if enabled)
    if normalize:
        # Find min and max values for each dimension of points
        min_vals = np.min(vertices, axis=0)
        max_vals = np.max(vertices, axis=0)
        
        # Calculate the center of the bounding box
        center = (min_vals + max_vals) / 2
        
        # Calculate the scale factor to normalize to [-1, 1]
        # We take the maximum range across all dimensions to preserve aspect ratio
        scale = np.max(max_vals - min_vals) / 2
        
        # Normalize points to [-1, 1]^3
        vertices = (vertices - center) / scale
        
        # Apply the same transformation to joints
        if joints is not None:
            joints = (joints - center) / scale
    
    # Update asset
    asset.vertices = vertices
    asset.joints = joints
    
    # Update matrix_local if it exists
    if asset.matrix_local is not None and joints is not None:
        asset.matrix_local[:, :3, 3] = joints


def pos_aug_transform(asset: Asset, 
              enable_pose_augmentation: bool = False,
              pose_angle_range: float = 0.0,
              pose_prob: float = 0.5,
              joint_specific_angles: bool = True):
    """
    Transform the asset data with comprehensive data augmentation including pose variations.
    
    Args:
        asset: Asset object to transform
        enable_rotation: Whether to apply random rotation
        enable_scaling: Whether to apply random scaling  
        enable_pose_augmentation: Whether to apply random pose variations
        rotation_range: Maximum rotation angle in degrees (applied to each axis)
        scaling_range: (min_scale, max_scale) for uniform scaling
        pose_angle_range: Maximum pose angle for joints in degrees
        pose_prob: Probability of applying pose augmentation
        joint_specific_angles: Whether to use joint-specific angle limits
        normalize: Whether to normalize to [-1, 1]^3
    """
    vertices = asset.vertices.copy()
    joints = asset.joints.copy() if asset.joints is not None else None
    
    # 1. Random Pose Augmentation (应用在最开始，因为会改变vertices和joints)
    if enable_pose_augmentation and joints is not None and np.random.rand() < pose_prob:
        if hasattr(asset, 'skin') and asset.skin is not None:
            # 应用随机姿态变换
            apply_random_pose_augmentation(
                asset, 
                pose_angle_range, 
                joint_specific_angles
            )
            vertices = asset.vertices.copy()
            joints = asset.joints.copy()
    
    
    # Update asset
    asset.vertices = vertices
    asset.joints = joints
    
    # Update matrix_local if it exists
    if asset.matrix_local is not None and joints is not None:
        asset.matrix_local[:, :3, 3] = joints

def apply_random_pose_augmentation(asset: Asset, pose_angle_range: float, joint_specific_angles: bool = True):
    """
    Apply random pose augmentation to the asset using matrix basis transformations.
    
    Args:
        asset: Asset object to transform
        pose_angle_range: Maximum pose angle in degrees
        joint_specific_angles: Whether to use joint-specific angle limits
    """
    if asset.joints is None or asset.skin is None:
        return
    try:
        if joint_specific_angles:
            # 为不同关节设置不同的角度限制
            joint_angle_limits = get_joint_angle_limits(asset.J)
            matrix_basis = get_constrained_random_pose(asset.J, joint_angle_limits, pose_angle_range)
        else:
            # 使用统一的角度限制
            matrix_basis = asset.get_random_matrix_basis(pose_angle_range)

        # 验证matrix_basis的维度
        if matrix_basis.shape != (asset.J, 4, 4):
            print(f"Warning: matrix_basis has wrong shape {matrix_basis.shape}, expected ({asset.J}, 4, 4)")
            return

        # 应用姿态变换
        asset.apply_matrix_basis(matrix_basis)
    except Exception as e:
        print(f"Warning: Failed to apply pose augmentation: {e}")
        # 如果失败，保持原始姿态
        pass

def get_joint_angle_limits(num_joints: int) -> Dict[int, float]:
    """
    Get joint-specific angle limits based on human anatomy.
    
    Args:
        num_joints: Number of joints
        
    Returns:
        Dictionary mapping joint index to angle limit in degrees
    """
    # 基于人体解剖学的关节角度限制
    joint_limits = {
        0: 15.0,   # hips - 髋部，较小的变化
        1: 20.0,   # spine - 脊柱
        2: 25.0,   # chest - 胸部
        3: 20.0,   # upper_chest - 上胸部
        4: 30.0,   # neck - 颈部，可以有较大变化
        5: 25.0,   # head - 头部
        6: 40.0,   # l_shoulder - 肩部，活动范围大
        7: 45.0,   # l_upper_arm - 上臂
        8: 60.0,   # l_lower_arm - 下臂，肘关节活动范围大
        9: 35.0,   # l_hand - 手部
        10: 40.0,  # r_shoulder
        11: 45.0,  # r_upper_arm
        12: 60.0,  # r_lower_arm
        13: 35.0,  # r_hand
        14: 40.0,  # l_upper_leg - 大腿
        15: 50.0,  # l_lower_leg - 小腿，膝关节活动范围大
        16: 30.0,  # l_foot - 脚部
        17: 20.0,  # l_toe_base - 脚趾
        18: 40.0,  # r_upper_leg
        19: 50.0,  # r_lower_leg
        20: 30.0,  # r_foot
        21: 20.0,  # r_toe_base
    }
    
    # 对于超出预定义范围的关节，使用默认值
    default_limit = 30.0
    result = {}
    for i in range(num_joints):
        result[i] = joint_limits.get(i, default_limit)
    
    return result

def get_constrained_random_pose(num_joints: int, 
                                joint_limits: Dict[int, float], 
                                base_angle_range: float,
                                norm: bool=False
                                ):
    """
    Generate random pose with joint-specific constraints.
    
    Args:
        num_joints: Number of joints
        joint_limits: Dictionary of joint angle limits
        base_angle_range: Base angle range to scale the limits
        
    Returns:
        matrix_basis: Random pose transformation matrices
    """
    
    # 为每个关节生成随机角度
    random_angles = np.zeros((num_joints, 3))
    
    for i in range(num_joints):
        joint_limit = joint_limits.get(i, 30.0)
        # 将基础角度范围和关节特定限制结合
        effective_limit = min(joint_limit, base_angle_range)
        
        if norm:
            # 生成正态分布的随机角度
            random_angles[i] = np.random.randn(3) * effective_limit / 180 * np.pi
        else:
            random_angles[i] = (np.random.rand(3) - 0.5) * 2 * effective_limit / 180 * np.pi

    
    # 转换为变换矩阵
    matrix_basis = axis_angle_to_matrix(random_angles).astype(np.float32)
    
    return matrix_basis

class RigDataset(Dataset):
    '''
    A simple dataset class.
    '''
    def __init__(
        self,
        data_root: str,
        paths: List[str],
        train: bool,
        batch_size: int,
        shuffle: bool,
        sampler: Sampler,
        transform: Union[Callable, None] = None,
        return_origin_vertices: bool = False,
        aug_prob: float = 0.0,
        rotation_range: float = 0.0,
        scaling_range: Tuple[float, float] = (1.0, 1.0),
        pose_angle_range: float = 0.0,
        track_pose_aug: bool = True,
        hand: bool = True,
    ):
        super().__init__()
        self.data_root  = data_root
        self.paths      = paths.copy()
        self.batch_size = batch_size
        self.train      = train
        self.shuffle    = shuffle
        self._sampler   = sampler # do not use `sampler` to avoid name conflict
        self.transform  = transform
        self.aug_rotation_range = rotation_range
        self.aug_scaling_range = scaling_range
        self.aug_prob = aug_prob
        self.pose_angle_range = pose_angle_range
        self.track_pose_aug = track_pose_aug
        self.hand = hand
        self.body_mask = body_mask
        
        self.return_origin_vertices = return_origin_vertices
        self.set_attrs(
            batch_size=self.batch_size,
            total_len=len(self.paths),
            shuffle=self.shuffle,
        )

        print('preparing dataset...')
        self.assets = {}
        for path in self.paths:
            asset = Asset.load(os.path.join(self.data_root, path))
            if self.transform is not None:
                self.transform(asset)
            self.assets[path] = asset

        if self.track_pose_aug:
            tracks_dir = 'dataB/track'
            self.matrix_basis = []
            for track in os.listdir(tracks_dir):
                if track.endswith('.npz'):
                    track_path = os.path.join(tracks_dir, track)
                    matrix_basis = np.load(track_path)['matrix_basis']
                    self.matrix_basis.append(matrix_basis)
            self.matrix_basis = np.concatenate(self.matrix_basis, axis=0)
    
    def __getitem__(self, index) -> Dict:
        """
        Get a sample from the dataset
        
        Args:
            index (int): Index of the sample
            
        Returns:
            data (Dict): Dictionary containing the following keys:
                - vertices: jt.Var, (B, N, 3) point cloud data
                - normals: jt.Var, (B, N, 3) point cloud normals
                - joints: jt.Var, (B, J, 3) joint positions
                - skin: jt.Var, (B, J, J) skinning weights
                - faces: jt.Var, (B, F, 3) face indices (if available)
        """
        
        path = self.paths[index]
        asset = copy.deepcopy(self.assets[path])

        if self.track_pose_aug:
            track_matrix = random.choice(self.matrix_basis)
            asset.apply_matrix_basis(track_matrix)

        if self.train:
            if self.aug_rotation_range > 0:
                if random.random() < self.aug_prob:
                    aug_transform(asset, 
                                enable_rotation=True, 
                                rotation_range=self.aug_rotation_range)
                    
            if self.aug_scaling_range[0] != 1.0 or self.aug_scaling_range[1] != 1.0:
                if random.random() < self.aug_prob:
                    aug_transform(asset, 
                                enable_scaling=True,
                                scaling_range=self.aug_scaling_range)
                    
            if self.pose_angle_range != 0:
                if random.random() < self.aug_prob:
                        pos_aug_transform(asset, 
                            enable_pose_augmentation=True,
                            pose_angle_range=self.pose_angle_range)

        origin_vertices = jt.array(asset.vertices.copy()).float32()
        
        sampled_asset = asset.sample(sampler=self._sampler)

        vertices    = jt.array(sampled_asset.vertices).float32()
        normals     = jt.array(sampled_asset.normals).float32()

        if sampled_asset.joints is not None:
            joints      = jt.array(sampled_asset.joints).float32()
        else:
            joints      = None

        if sampled_asset.skin is not None:
            skin        = jt.array(sampled_asset.skin).float32()
        else:
            skin        = None

        res = {
            'vertices': vertices,
            'normals': normals,
            'cls': asset.cls,
            'id': asset.id,
        }
        if joints is not None:
            res['joints'] = joints if self.hand else joints[:22]
        if skin is not None:
            res['skin'] = skin
        if self.return_origin_vertices:
            res['origin_vertices'] = origin_vertices
            
        # # 添加面片信息（如果可用）
        # if hasattr(asset, 'faces') and asset.faces is not None:
        #     res['faces'] = jt.array(asset.faces).long()
        #     res['origin_faces'] = jt.array(asset.faces).long()
            
        return res
    
    def collate_batch(self, batch):
        if self.return_origin_vertices:
            max_N = 0
            for b in batch:
                max_N = max(max_N, b['origin_vertices'].shape[0])
            for b in batch:
                N = b['origin_vertices'].shape[0]
                b['origin_vertices'] = np.pad(b['origin_vertices'], ((0, max_N-N), (0, 0)), 'constant', constant_values=0.)
                b['N'] = N
        
        # # 处理面片信息的批次整理
        # if any('faces' in b for b in batch):
        #     # 找到最大面片数量
        #     max_F = 0
        #     for b in batch:
        #         if 'faces' in b:
        #             max_F = max(max_F, b['faces'].shape[0])
            
        #     # 填充面片数组
        #     for b in batch:
        #         if 'faces' in b:
        #             F = b['faces'].shape[0]
        #             if F < max_F:
        #                 # 用最后一个面片重复填充
        #                 last_face = b['faces'][-1:].repeat(max_F - F, 0)
        #                 b['faces'] = np.concatenate([b['faces'], last_face], axis=0)
        #             b['F'] = F
        #         else:
        #             # 如果没有面片信息，创建虚拟面片
        #             b['faces'] = np.zeros((max_F, 3), dtype=np.int64)
        #             b['F'] = 0
                    
        #     # 同样处理origin_faces
        #     if any('origin_faces' in b for b in batch):
        #         for b in batch:
        #             if 'origin_faces' in b:
        #                 F = b['origin_faces'].shape[0]
        #                 if F < max_F:
        #                     last_face = b['origin_faces'][-1:].repeat(max_F - F, 0)
        #                     b['origin_faces'] = np.concatenate([b['origin_faces'], last_face], axis=0)
        #             else:
        #                 b['origin_faces'] = np.zeros((max_F, 3), dtype=np.int64)
        
        return super().collate_batch(batch)

# Example usage of the dataset
def get_dataloader(
    data_root: str,
    data_list: str,
    hand: bool,
    train: bool,
    batch_size: int,
    shuffle: bool,
    sampler: Sampler,
    transform: Union[Callable, None] = None,
    return_origin_vertices: bool = False,
    rotation_range: float = 0.0,
    scaling_range: Tuple[float, float] = (1.0, 1.0),
    aug_prob: float = 0.0,
    pose_angle_range: float = 0.0,
    track_pose_aug: bool = True,
    drop_bad: bool = False,
):
    """
    Create a dataloader for point cloud data
    
    Args:
        data_root (str): Root directory for the data files
        data_list (str): Path to the file containing list of data files
        train (bool): Whether the dataset is for training
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the dataset
        sampler (Sampler): Sampler to use for point cloud sampling
        transform (callable, optional): Optional post-transform to be applied on a sample
        return_origin_vertices (bool): Whether to return original vertices
        
    Returns:
        dataset (RigDataset): The dataset
    """
    drop_list = []
    if drop_bad:
        with open('drop_list.txt', 'r') as drop:
            drop_list = [line.strip() for line in drop.readlines()]
        
    with open(data_list, 'r') as f:
        paths = [line.strip() for line in f.readlines()]
        paths = [p for p in paths if p not in drop_list]
    
    dataset = RigDataset(
        data_root=data_root,
        paths=paths,
        hand=hand,
        train=train,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        transform=transform,
        return_origin_vertices=return_origin_vertices,
        rotation_range=rotation_range,
        scaling_range=scaling_range,
        aug_prob=aug_prob,
        pose_angle_range=pose_angle_range,
        track_pose_aug=track_pose_aug,
    )
    
    return dataset
