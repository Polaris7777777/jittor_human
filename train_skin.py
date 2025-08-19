import jittor as jt
import numpy as np
import os
import shutil
import argparse
import time
import random

from jittor import nn
from jittor import optim, lr_scheduler

from dataset.dataset import get_dataloader, transform
from dataset.format import id_to_name
from dataset.sampler import SamplerMix
from models.skin import create_model

from dataset.exporter import Exporter

# Set Jittor flags
jt.flags.use_cuda = 1

def add_joint_noise(joints, noise_type='gaussian', noise_scale=0.01, noise_prob=0.5):
    """
    为关节点添加随机扰动
    
    Args:
        joints: [B, J, 3] 关节点坐标
        noise_type: str, 噪声类型 ('gaussian', 'uniform', 'outlier')
        noise_scale: float, 噪声强度
        noise_prob: float, 应用噪声的概率
        
    Returns:
        noisy_joints: [B, J, 3] 添加噪声后的关节点
    """
    if np.random.rand() > noise_prob:
        return joints
    
    batch_size, num_joints, dim = joints.shape
    
    if noise_type == 'gaussian':
        # 高斯噪声
        noise = jt.randn_like(joints) * noise_scale
        noisy_joints = joints + noise
        
    elif noise_type == 'uniform':
        # 均匀噪声
        noise = (jt.rand_like(joints) - 0.5) * 2 * noise_scale
        noisy_joints = joints + noise
        
    elif noise_type == 'outlier':
        # 离群点噪声 - 随机选择少数关节添加大噪声
        noisy_joints = joints.clone()
        num_outliers = max(1, int(num_joints * 0.1))  # 10%的关节作为离群点
        
        for b in range(batch_size):
            outlier_indices = np.random.choice(num_joints, num_outliers, replace=False)
            outlier_noise = jt.randn(num_outliers, dim) * noise_scale * 5  # 5倍的噪声
            noisy_joints[b, outlier_indices] += outlier_noise
            
    elif noise_type == 'mixed':
        # 混合噪声
        # 50% 高斯噪声 + 30% 均匀噪声 + 20% 无噪声
        rand_val = np.random.rand()
        if rand_val < 0.5:
            noise = jt.randn_like(joints) * noise_scale
        elif rand_val < 0.8:
            noise = (jt.rand_like(joints) - 0.5) * 2 * noise_scale
        else:
            noise = jt.zeros_like(joints)
        noisy_joints = joints + noise
        
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return noisy_joints

def add_adaptive_joint_noise(joints, noise_scale=0.01, bone_aware=True):
    """
    自适应关节噪声 - 根据关节的重要性调整噪声强度
    
    Args:
        joints: [B, J, 3] 关节点坐标
        noise_scale: float, 基础噪声强度
        bone_aware: bool, 是否考虑骨骼结构
        
    Returns:
        noisy_joints: [B, J, 3] 添加自适应噪声后的关节点
    """
    batch_size, num_joints, dim = joints.shape
    
    # 定义关节重要性权重（基于人体骨骼结构）
    joint_importance = {
        0: 1.0,   # hips - 非常重要
        1: 0.8,   # spine
        2: 0.8,   # chest
        3: 0.6,   # upper_chest
        4: 0.4,   # neck
        5: 0.3,   # head
        6: 0.7,   # l_shoulder
        7: 0.6,   # l_upper_arm
        8: 0.5,   # l_lower_arm
        9: 0.3,   # l_hand
        10: 0.7,  # r_shoulder
        11: 0.6,  # r_upper_arm
        12: 0.5,  # r_lower_arm
        13: 0.3,  # r_hand
        14: 0.8,  # l_upper_leg
        15: 0.7,  # l_lower_leg
        16: 0.5,  # l_foot
        17: 0.2,  # l_toe_base
        18: 0.8,  # r_upper_leg
        19: 0.7,  # r_lower_leg
        20: 0.5,  # r_foot
        21: 0.2,  # r_toe_base
    }
    
    noisy_joints = joints.clone()
    
    for joint_idx in range(min(num_joints, len(joint_importance))):
        # 重要性越高，噪声越小
        importance = joint_importance.get(joint_idx, 0.5)
        adaptive_scale = noise_scale * (2.0 - importance)  # 重要关节噪声更小
        
        # 生成自适应噪声
        noise = jt.randn(batch_size, dim) * adaptive_scale
        noisy_joints[:, joint_idx] += noise
    
    return noisy_joints

def add_hierarchical_joint_noise(joints, parents, noise_scale=0.01):
    """
    层次化关节噪声 - 考虑父子关节关系的噪声传播
    
    Args:
        joints: [B, J, 3] 关节点坐标
        parents: List[int], 父关节索引
        noise_scale: float, 噪声强度
        
    Returns:
        noisy_joints: [B, J, 3] 添加层次化噪声后的关节点
    """
    from dataset.format import parents as default_parents
    
    if parents is None:
        parents = default_parents
    
    batch_size, num_joints, dim = joints.shape
    noisy_joints = joints.clone()
    
    # 为根关节添加噪声
    root_noise = jt.randn(batch_size, dim) * noise_scale
    noisy_joints[:, 0] += root_noise  # 假设第0个是根关节
    
    # 层次化传播噪声
    for joint_idx in range(1, num_joints):
        parent_idx = parents[joint_idx]
        if parent_idx is not None and parent_idx < num_joints:
            # 父关节的噪声会影响子关节
            parent_noise_influence = 0.3  # 父关节噪声的影响程度
            inherited_noise = (noisy_joints[:, parent_idx] - joints[:, parent_idx]) * parent_noise_influence
            
            # 自身噪声
            own_noise = jt.randn(batch_size, dim) * noise_scale * 0.7
            
            # 组合噪声
            total_noise = inherited_noise + own_noise
            noisy_joints[:, joint_idx] += total_noise
        else:
            # 独立噪声
            own_noise = jt.randn(batch_size, dim) * noise_scale
            noisy_joints[:, joint_idx] += own_noise
    
    return noisy_joints

def train(args):
    """
    Main training function with joint noise augmentation
    
    Args:
        args: Command line arguments
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set up logging
    log_file = os.path.join(args.output_dir, 'training_log.txt')
    vital_files = ['train_skin.py', 'models/skin.py', 'dataset/dataset.py']
    for file in vital_files:
        shutil.copy(file, args.output_dir)
    def log_message(message):
        """Helper function to log messages to file and print to console"""
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")
        print(message)
    
    # Log training parameters
    log_message(f"Starting training with parameters: {args}")
    log_message(f"Joint noise enabled: {args.enable_joint_noise}")
    if args.enable_joint_noise:
        log_message(f"Joint noise type: {args.joint_noise_type}")
        log_message(f"Joint noise scale: {args.joint_noise_scale}")
        log_message(f"Joint noise probability: {args.joint_noise_prob}")
    
    # Create model
    model = create_model(
        model_name=args.model_name,
        num_joints=52,
        feat_dim=args.feat_dim,
        pct_feat_dim=args.pct_feat_dim
    )
    
    # Load pre-trained model if specified
    if args.pretrained_model:
        log_message(f"Loading pretrained model from {args.pretrained_model}")
        model.load(args.pretrained_model)
    
    # Create optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # create lr scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_decay)

    # Create loss function
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    
    # Create dataloaders
    train_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.train_data_list,
        train=True,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=SamplerMix(num_samples=args.num_samples, vertex_samples=args.vertex_samples),
        transform=transform,
        aug_prob=args.aug_prob,
        rotation_range=args.rotation_range,
        scaling_range=args.scaling_range,
        pose_angle_range=args.pose_angle_range,
        track_pose_aug=args.track_pose_aug,
        drop_bad=args.drop_bad
    )
    
    if args.val_data_list:
        val_loader = get_dataloader(
            data_root=args.data_root,
            data_list=args.val_data_list,
            train=False,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=SamplerMix(num_samples=args.num_samples, vertex_samples=args.vertex_samples),
            transform=transform,
            pose_angle_range=0.0,
            track_pose_aug=args.track_pose_aug,
        )
    else:
        val_loader = None
    
    # Training loop
    best_loss = 99999999
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss_mse = 0.0
        train_loss_l1 = 0.0
        
        start_time = time.time()
        for batch_idx, data in enumerate(train_loader):
            # Get data and labels
            vertices, joints, skin = data['vertices'], data['joints'], data['skin']

            # Apply joint noise augmentation during training
            if args.enable_joint_noise:
                if args.joint_noise_type == 'adaptive':
                    joints = add_adaptive_joint_noise(
                        joints, 
                        noise_scale=args.joint_noise_scale
                    )
                elif args.joint_noise_type == 'hierarchical':
                    joints = add_hierarchical_joint_noise(
                        joints, 
                        parents=None,  # 使用默认父关节关系
                        noise_scale=args.joint_noise_scale
                    )
                else:
                    joints = add_joint_noise(
                        joints,
                        noise_type=args.joint_noise_type,
                        noise_scale=args.joint_noise_scale,
                        noise_prob=args.joint_noise_prob
                    )

            vertices: jt.Var
            joints: jt.Var
            skin: jt.Var
            outputs = model(vertices, joints)
            loss_mse = criterion_mse(outputs, skin)
            loss_l1 = criterion_l1(outputs, skin)
            loss = loss_mse + loss_l1
            
            # Backward pass and optimize
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            
            # Calculate statistics
            train_loss_mse += loss_mse.item()
            train_loss_l1 += loss_l1.item()
            
            # Print progress
            if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(train_loader):
                log_message(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                           f"Loss mse: {loss_mse.item():.4f} Loss l1: {loss_l1.item():.4f}")
                
        # Step the scheduler
        scheduler.step()
        
        # Calculate epoch statistics
        train_loss_mse /= len(train_loader)
        train_loss_l1 /= len(train_loader)
        epoch_time = time.time() - start_time
        
        log_message(f"Epoch [{epoch+1}/{args.epochs}] "
                   f"Train Loss mse: {train_loss_mse:.4f} "
                   f"Train Loss l1: {train_loss_l1:.4f} "
                   f"Time: {epoch_time:.2f}s "
                   f"LR: {optimizer.lr:.6f}")

        # Validation phase (不在验证时添加噪声)
        if val_loader is not None and (epoch + 1) % args.val_freq == 0:
            model.eval()
            val_loss_mse = 0.0
            val_loss_l1 = 0.0
            
            show_id = np.random.randint(0, len(val_loader))
            for batch_idx, data in enumerate(val_loader):
                # Get data and labels (不添加噪声)
                vertices, joints, skin = data['vertices'], data['joints'], data['skin']
                
                # Forward pass
                outputs = model(vertices, joints)
                loss_mse = criterion_mse(outputs, skin)
                loss_l1 = criterion_l1(outputs, skin)
                
                # export render results(which is slow, so you can turn it off)
                if batch_idx == show_id:
                    exporter = Exporter()
                    for i in id_to_name:
                        name = id_to_name[i]
                        # export every joint's corresponding skinning
                        exporter._render_skin(path=f"{args.output_dir}/tmp/skin/epoch_{epoch}/{name}_ref.png",vertices=vertices.numpy()[0], skin=skin.numpy()[0, :, i], joint=joints[0, i])
                        exporter._render_skin(path=f"{args.output_dir}/tmp/skin/epoch_{epoch}/{name}_pred.png",vertices=vertices.numpy()[0], skin=outputs.numpy()[0, :, i], joint=joints[0, i])

                val_loss_mse += loss_mse.item()
                val_loss_l1 += loss_l1.item()
            
            # Calculate validation statistics
            val_loss_mse /= len(val_loader)
            val_loss_l1 /= len(val_loader)
            
            log_message(f"Validation Loss: mse: {val_loss_mse:.4f} l1: {val_loss_l1:.4f}")
            
            # Save best model
            if val_loss_l1 < best_loss:
                best_loss = val_loss_l1
                model_path = os.path.join(args.output_dir, 'best_model.pkl')
                model.save(model_path)
                log_message(f"Saved best model with l1 loss {best_loss:.4f} to {model_path}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pkl')
            model.save(checkpoint_path)
            log_message(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pkl')
    model.save(final_model_path)
    log_message(f"Training completed. Saved final model to {final_model_path}")
    
    return model, best_loss

def main():
    """Parse arguments and start training"""
    parser = argparse.ArgumentParser(description='Train a point cloud model')
    
    # Dataset parameters
    parser.add_argument('--train_data_list', type=str, required=True,
                        help='Path to the training data list file')
    parser.add_argument('--val_data_list', type=str, default='',
                        help='Path to the validation data list file')
    parser.add_argument('--data_root', type=str, default='data',
                        help='Root directory for the data files')
    parser.add_argument('--num_samples', type=int, default=1024,
                        help='Number of samples per point cloud')
    parser.add_argument('--vertex_samples', type=int, default=512,
                        help='Number of vertex samples for skinning')
    parser.add_argument('--aug_prob', type=float, default=0.0,
                        help='Probability of applying augmentation')
    parser.add_argument('--rotation_range', type=float, default=0.0,
                        help='Rotation range for augmentation')
    parser.add_argument('--scaling_range', type=float, nargs=2, default=(1.0, 1.0),
                        help='Scaling range for augmentation')
    parser.add_argument('--drop_bad', action='store_true', default=False,
                        help='Drop bad samples during training')
    parser.add_argument('--pose_angle_range', type=float, default=0.0,
                        help='Pose angle range for augmentation')
    parser.add_argument('--track_pose_aug', action='store_true', default=False,
                        help='Whether to apply pose tracking augmentation')
    # Model parameters
    parser.add_argument('--model_name', type=str, default='pct',
                        choices=['pct', 'pct2', 'custom_pct', 'skeleton'],
                        help='Model architecture to use')
    parser.add_argument('--model_type', type=str, default='standard',
                        choices=['standard', 'enhanced'],
                        help='Model type for skeleton model')
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='Path to pretrained model')
    parser.add_argument('--feat_dim', type=int, default=256,
                        help='Feature dimension for the model')
    parser.add_argument('--pct_feat_dim', type=int, default=128,
                        help='Feature dimension for PCT model')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['sgd', 'adamw'],
                        help='Optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--lr_step', type=int, default=20,
                        help='Step size for learning rate decay')
    parser.add_argument('--lr_decay', type=float, default=0.8,
                        help='Decay factor for learning rate')
    
    # Joint noise augmentation parameters
    parser.add_argument('--enable_joint_noise', action='store_true', default=False,
                        help='Enable joint noise augmentation')
    parser.add_argument('--joint_noise_type', type=str, default='gaussian',
                        choices=['gaussian', 'uniform', 'outlier', 'mixed', 'adaptive', 'hierarchical'],
                        help='Type of joint noise to apply')
    parser.add_argument('--joint_noise_scale', type=float, default=0.01,
                        help='Scale of joint noise')
    parser.add_argument('--joint_noise_prob', type=float, default=0.5,
                        help='Probability of applying joint noise')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='output/skin',
                        help='Directory to save output files')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save frequency')
    parser.add_argument('--val_freq', type=int, default=1,
                        help='Validation frequency')
    
    args = parser.parse_args()
    
    # Start training
    train(args)

def seed_all(seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    seed_all(123)
    main()