import jittor as jt
import numpy as np
import os
import shutil
import argparse
import time
import random

from jittor import nn
from jittor import optim, lr_scheduler
from jittor.dataset import Dataset

from dataset.exporter import Exporter
from models.codebook import create_keypoint_vqvae
from models.pose_aware_codebook import create_pose_aware_keypoint_vqvae

from models.metrics import (
    J2J, bone_length_symmetry_loss, symmetry_loss, 
    topology_loss, relative_position_loss
)

import wandb
os.environ["WANDB_MODE"] = "offline"

# if hard (B榜任务)
dataset_with_hand = os.environ.get('HARD', '0').lower() in ['1', 'true', 'yes', 'on']
print(f'Joints: {52 if dataset_with_hand else 22}')

# Set Jittor flags
jt.flags.use_cuda = 1

class KeypointDataset(Dataset):
    """Simple dataset for loading keypoint data only"""
    
    def __init__(self, data_root, data_list, hand=False, train=True, 
                 rotation_range=0, scaling_range=(1.0, 1.0), aug_prob=0.0, 
                 pose_angle_range=0, track_pose_aug=False):
        super().__init__()
        self.data_root = data_root
        self.hand = hand
        self.train = train
        self.rotation_range = rotation_range
        self.scaling_range = scaling_range
        self.aug_prob = aug_prob
        self.pose_angle_range = pose_angle_range
        self.track_pose_aug = track_pose_aug
        
        # Load data list
        with open(data_list, 'r') as f:
            self.data_files = [line.strip() for line in f.readlines()]
        
        # Load motion capture data if track_pose_aug is enabled
        self.matrix_basis = []
        if self.track_pose_aug and train:
            tracks_dir = os.path.join(data_root, 'track')
            if os.path.exists(tracks_dir):
                print(f"Loading motion capture data from {tracks_dir}")
                for track_file in os.listdir(tracks_dir):
                    if track_file.endswith('.npz'):
                        track_path = os.path.join(tracks_dir, track_file)
                        try:
                            track_data = np.load(track_path)
                            if 'matrix_basis' in track_data:
                                matrix_basis = track_data['matrix_basis']
                                self.matrix_basis.append(matrix_basis)
                                print(f"Loaded {matrix_basis.shape[0]} frames from {track_file}")
                        except Exception as e:
                            print(f"Warning: Failed to load {track_file}: {e}")
                
                if self.matrix_basis:
                    self.matrix_basis = np.concatenate(self.matrix_basis, axis=0)
                    print(f"Total motion capture frames: {self.matrix_basis.shape[0]}")
                else:
                    print("Warning: No valid motion capture data found")
                    self.track_pose_aug = False
            else:
                print(f"Warning: Track directory {tracks_dir} not found")
                self.track_pose_aug = False
        
        print(f"Loaded {len(self.data_files)} samples from {data_list}")
        print(f"Expected {52 if hand else 22} joints per sample")
        print(f"Track pose augmentation: {'Enabled' if self.track_pose_aug else 'Disabled'}")
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        # Load the data file
        data_path = os.path.join(self.data_root, self.data_files[idx])
        data = np.load(data_path)
        
        # Extract joints
        if 'joints' in data:
            joints = data['joints'].astype(np.float32)
        elif 'joint' in data:
            joints = data['joint'].astype(np.float32)
        else:
            # Fallback to look for any joint-related key
            joint_keys = [k for k in data.keys() if 'joint' in k.lower()]
            if joint_keys:
                joints = data[joint_keys[0]].astype(np.float32)
            else:
                raise ValueError(f"No joint data found in {data_path}")
        
        # Ensure correct shape [J, 3]
        if joints.ndim == 1:
            joints = joints.reshape(-1, 3)
        
        # Verify joint number matches expectation
        expected_num_joints = 52 if self.hand else 22
        actual_num_joints = joints.shape[0]
        
        if actual_num_joints != expected_num_joints:
            raise ValueError(f"Joint number mismatch in {data_path}: "
                           f"expected {expected_num_joints}, got {actual_num_joints}")
        
        # Apply data augmentation if training
        if self.train and np.random.rand() < self.aug_prob:
            joints = self._augment_joints(joints)
        
        # Apply motion capture pose augmentation
        if self.train and self.track_pose_aug and len(self.matrix_basis) > 0:
            if np.random.rand() < 0.3:  # 30% chance to apply motion capture pose
                joints = self._apply_mocap_pose(joints, data_path)
        
        return {
            'joints': jt.array(joints)
        }
    
    def _augment_joints(self, joints):
        """Apply basic data augmentation to joints"""
        # Random rotation around Y axis
        if self.rotation_range > 0:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot_y = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
            joints = joints @ rot_y.T
        
        # Random scaling
        if self.scaling_range[0] != self.scaling_range[1]:
            scale = np.random.uniform(self.scaling_range[0], self.scaling_range[1])
            joints = joints * scale
        
        return joints
    
    def _apply_mocap_pose(self, joints, data_path):
        """Apply motion capture pose augmentation"""
        try:
            # Load the original asset data for pose transformation
            from dataset.asset import Asset
            
            # Load asset
            asset = Asset.load(data_path)
            
            # Randomly select a motion capture frame
            frame_idx = np.random.randint(0, self.matrix_basis.shape[0])
            matrix_basis = self.matrix_basis[frame_idx]
            
            # Apply the motion capture pose
            if hasattr(asset, 'apply_matrix_basis') and matrix_basis.shape[0] >= joints.shape[0]:
                # Use only the required number of joints
                matrix_basis_subset = matrix_basis[:joints.shape[0]]
                asset.apply_matrix_basis(matrix_basis_subset)
                
                # Return the transformed joints
                if asset.joints is not None:
                    return asset.joints.astype(np.float32)
        
        except Exception as e:
            print(f"Warning: Failed to apply motion capture pose: {e}")
        
        # Return original joints if motion capture augmentation fails
        return joints

def train(args):
    """
    Main training function for keypoint codebook
    
    Args:
        args: Command line arguments
    """
    if jt.rank == 0:
        # Create output directory if it doesn't exist
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            
        # Create directory for temporary outputs
        tmp_dir = os.path.join(args.output_dir, 'tmp', 'codebook')
        os.makedirs(tmp_dir, exist_ok=True)
        
        # Set up logging
        log_file = os.path.join(args.output_dir, 'training_log.txt')
        vital_files = ['train_codebook.py', 'models/codebook.py']
        for file in vital_files:
            if os.path.exists(file):
                shutil.copy(file, args.output_dir)
            
        wandb.init(project="jittor-keypoint-codebook", config=args, name=f"codebook_{args.model_type}", dir=args.output_dir)
        wandb.run.name = f"codebook_{time.strftime('%Y%m%d_%H%M%S')}"

        def log_message(message):
            """Helper function to log messages to file and print to console"""
            with open(log_file, 'a') as f:
                f.write(f"{message}\n")
            print(message)
        
        # Log training parameters
        log_message(f"Starting keypoint codebook training with parameters: {args}")
    
    # Create model with correct joint number
    num_joints = 52 if dataset_with_hand else 22
    print(f"Creating model for {num_joints} joints")
    
    # Choose model type based on args
    if args.model_type == 'pose_aware_vqvae':
        model = create_pose_aware_keypoint_vqvae(
            num_joints=num_joints,
            embedding_dim=args.embedding_dim,
            num_embeddings=args.num_embeddings,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            commitment_cost=args.commitment_cost
        )
    else:
        # Default to standard transformer codebook
        model = create_keypoint_vqvae(
            num_joints=num_joints,
            embedding_dim=args.embedding_dim,
            num_embeddings=args.num_embeddings,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            commitment_cost=args.commitment_cost
        )

    # Load pre-trained model if specified
    if args.pretrained_model:
        if jt.rank==0: 
            log_message(f"Loading pretrained model from {args.pretrained_model}")
        model.load(args.pretrained_model)
    
    if jt.rank==0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log_message(f"Total trainable parameters: {total_params}")

    # Create optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Create lr scheduler
    if args.lr_scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_decay)
    elif args.lr_scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)

    # Create loss function for joint reconstruction
    criterion_joint = nn.MSELoss()
    
    # Create datasets - direct keypoint loading
    train_dataset = KeypointDataset(
        data_root=args.data_root,
        data_list=args.train_data_list,
        hand=dataset_with_hand,
        train=True,
        rotation_range=args.rotation_range,
        scaling_range=args.scaling_range,
        aug_prob=args.aug_prob,
        pose_angle_range=args.pose_angle_range,
        track_pose_aug=args.track_pose_aug
    )
    
    train_loader = train_dataset.set_attrs(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    if args.val_data_list:
        val_dataset = KeypointDataset(
            data_root=args.data_root,
            data_list=args.val_data_list,
            hand=dataset_with_hand,
            train=False,
            track_pose_aug=False  # No augmentation for validation
        )
        val_loader = val_dataset.set_attrs(
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )
    else:
        val_loader = None
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        # Training phase
        model.train()

        if jt.rank==0:
            train_loss = 0.0
            loss_dict = {
                'joint_mse': 0.0,
                'vq_loss': 0.0,
                'J2J': 0.0,
                'symmetry': 0.0,
                'bone_length_symmetry': 0.0,
                'topology': 0.0,
                'relative_position': 0.0,
                'perplexity': 0.0
            }
            start_time = time.time()
        
        for batch_idx, data in enumerate(train_loader):
            # Get keypoint data only
            joints = data['joints']      # [B, J, 3]
            
            # Forward pass - directly encode and decode keypoints
            outputs, vq_loss, indices, perplexity = model(joints, return_indices=True)
            
            # Joint reconstruction loss
            joint_mse_loss = criterion_joint(outputs, joints)
            
            # Additional losses for better bone structure - compute for each sample in batch
            batch_size = outputs.shape[0]
            # Fix: use simple scalar initialization instead of squeeze
            J2J_loss = jt.array(0.0)
            sym_loss = jt.array(0.0)
            sym_bone_loss = jt.array(0.0)
            topo_loss = jt.array(0.0)
            rel_pos_loss = jt.array(0.0)
            
            for i in range(batch_size):
                # Add batch dimension for compatibility with metric functions
                output_single = outputs[i:i+1]  # [1, J, 3]
                joint_single = joints[i:i+1]    # [1, J, 3]
                
                J2J_loss += J2J(outputs[i], joints[i]) / batch_size
                sym_loss += symmetry_loss(output_single, joint_single) / batch_size
                sym_bone_loss += bone_length_symmetry_loss(output_single) / batch_size
                topo_loss += topology_loss(output_single, joint_single) / batch_size
                rel_pos_loss += relative_position_loss(output_single, joint_single) / batch_size

            # Total loss
            loss = (
                joint_mse_loss +
                vq_loss * args.vq_loss_weight +
                J2J_loss * args.J2J_loss_weight +
                sym_loss * args.sym_loss_weight +
                sym_bone_loss * args.bone_length_symmetry_weight +
                topo_loss * args.topo_loss_weight +
                rel_pos_loss * args.rel_pos_loss_weight
            )

            # Backward pass and optimize
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            
            # Step the scheduler every iteration
            scheduler.step()

            if jt.in_mpi:
                # Gather losses from all processes
                loss = loss.mpi_all_reduce()
                joint_mse_loss = joint_mse_loss.mpi_all_reduce()
                vq_loss = vq_loss.mpi_all_reduce()
                J2J_loss = J2J_loss.mpi_all_reduce()
                sym_loss = sym_loss.mpi_all_reduce()
                sym_bone_loss = sym_bone_loss.mpi_all_reduce()
                topo_loss = topo_loss.mpi_all_reduce()
                rel_pos_loss = rel_pos_loss.mpi_all_reduce()
                perplexity = perplexity.mpi_all_reduce()

            if jt.rank==0:
                # Calculate statistics
                train_loss += loss.item()

                # Update loss statistics
                loss_dict['joint_mse'] += joint_mse_loss.item()
                loss_dict['vq_loss'] += vq_loss.item()
                loss_dict['J2J'] += J2J_loss.item()
                loss_dict['symmetry'] += sym_loss.item()
                loss_dict['bone_length_symmetry'] += sym_bone_loss.item()
                loss_dict['topology'] += topo_loss.item()
                loss_dict['relative_position'] += rel_pos_loss.item()
                loss_dict['perplexity'] += perplexity.item()
                
                # Print progress
                if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(train_loader):
                    log_message(
                        f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                        f"Train Loss: {train_loss/(batch_idx+1):.4f} "
                        f"MSE Loss: {loss_dict['joint_mse']/(batch_idx+1):.4f} "
                        f"VQ Loss: {loss_dict['vq_loss']/(batch_idx+1):.4f} "
                        f"J2J Loss: {loss_dict['J2J']/(batch_idx+1):.6f} "
                        f"Perplexity: {loss_dict['perplexity']/(batch_idx+1):.1f}"
                    )
                    
                    global_step = epoch * len(train_loader) + batch_idx
                    wandb.log({
                        "epoch": epoch + 1,
                        "total_loss": loss.item(),
                        "joint_mse_loss": loss_dict['joint_mse'] / (batch_idx + 1),
                        "vq_loss": loss_dict['vq_loss'] / (batch_idx + 1),
                        "J2J_loss": loss_dict['J2J'] / (batch_idx + 1),
                        "symmetry_loss": loss_dict['symmetry'] / (batch_idx + 1),
                        "topology_loss": loss_dict['topology'] / (batch_idx + 1),
                        "relative_position_loss": loss_dict['relative_position'] / (batch_idx + 1),
                        "perplexity": loss_dict['perplexity'] / (batch_idx + 1),
                        "learning_rate": optimizer.lr if hasattr(optimizer, 'lr') else args.learning_rate
                    }, step=global_step)
        
        if jt.rank==0:
            # Calculate epoch statistics
            train_loss /= len(train_loader)
            epoch_time = time.time() - start_time
            
            log_message(
                f"Epoch [{epoch+1}/{args.epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Time: {epoch_time:.2f}s "
                f"LR: {optimizer.lr:.6f}"
            )
            
            # Detailed loss report
            log_message(
                f"Detailed losses - "
                f"MSE: {loss_dict['joint_mse']/len(train_loader):.4f}, "
                f"VQ: {loss_dict['vq_loss']/len(train_loader):.4f}, "
                f"J2J: {loss_dict['J2J']/len(train_loader):.6f}, "
                f"Sym: {loss_dict['symmetry']/len(train_loader):.4f}, "
                f"BoneSym: {loss_dict['bone_length_symmetry']/len(train_loader):.4f}, "
                f"Topo: {loss_dict['topology']/len(train_loader):.4f}, "
                f"RelPos: {loss_dict['relative_position']/len(train_loader):.4f}"
            )
        
        # Validation phase
        if val_loader is not None and (epoch + 1) % args.val_freq == 0:
            model.eval()
            if jt.rank == 0:
                val_loss = 0.0
                val_J2J_loss = 0.0
                val_vq_loss = 0.0
                val_perplexity = 0.0
                show_id = np.random.randint(0, len(val_loader))
                
            for batch_idx, data in enumerate(val_loader):
                joints = data['joints']
                
                # Forward pass
                outputs, vq_loss, indices, perplexity = model(joints, return_indices=True)
                    
                loss = criterion_joint(outputs, joints)
                
                if jt.in_mpi:
                    loss = loss.mpi_all_reduce()
                    vq_loss = vq_loss.mpi_all_reduce()
                    perplexity = perplexity.mpi_all_reduce()

                if jt.rank == 0:
                    # Export render results for visualization
                    if batch_idx == show_id:
                        exporter = Exporter()
                        from dataset.format import parents
                        os.makedirs(f"{args.output_dir}/tmp/codebook/epoch_{epoch+1}", exist_ok=True)
                        
                        exporter._render_skeleton(
                            path=f"{args.output_dir}/tmp/codebook/epoch_{epoch+1}/skeleton_ref.png", 
                            joints=joints[0].numpy().reshape(-1, 3), 
                            parents=parents
                        )
                        exporter._render_skeleton(
                            path=f"{args.output_dir}/tmp/codebook/epoch_{epoch+1}/skeleton_pred.png", 
                            joints=outputs[0].numpy().reshape(-1, 3), 
                            parents=parents
                        )
                        
                        # Visualize codebook usage distribution
                        if indices is not None:
                            import matplotlib.pyplot as plt
                            index_counts = np.zeros(args.num_embeddings)
                            indices_np = indices.numpy() if hasattr(indices, 'numpy') else indices.data
                            for idx in indices_np.flatten():
                                if 0 <= idx < args.num_embeddings:
                                    index_counts[idx] += 1
                                
                            plt.figure(figsize=(10, 5))
                            plt.bar(range(args.num_embeddings), index_counts)
                            plt.title(f"Codebook Usage (Perplexity: {perplexity.item():.1f})")
                            plt.xlabel("Codebook Index")
                            plt.ylabel("Count")
                            plt.savefig(f"{args.output_dir}/tmp/codebook/epoch_{epoch+1}/codebook_usage.png")
                            plt.close()

                    val_loss += loss.item()
                    val_vq_loss += vq_loss.item()
                    val_perplexity += perplexity.item()
                    # Fix: compute J2J loss for each sample in batch
                    batch_J2J_loss = 0.0
                    for i in range(outputs.shape[0]):
                        batch_J2J_loss += J2J(outputs[i], joints[i]).item() / outputs.shape[0]
                    val_J2J_loss += batch_J2J_loss
            
            if jt.rank == 0:
                val_loss /= len(val_loader)
                val_vq_loss /= len(val_loader)
                val_J2J_loss /= len(val_loader)
                val_perplexity /= len(val_loader)
                
                log_message(
                    f"Validation - "
                    f"Loss: {val_loss:.4f}, "
                    f"VQ Loss: {val_vq_loss:.4f}, "
                    f"J2J Loss: {val_J2J_loss:.6f}, "
                    f"Perplexity: {val_perplexity:.1f}"
                )
                
                wandb.log({
                    "val_loss": val_loss,
                    "val_vq_loss": val_vq_loss,
                    "val_J2J_loss": val_J2J_loss,
                    "val_perplexity": val_perplexity,
                    "epoch": epoch + 1
                })

                # Save best model
                if val_J2J_loss < best_loss:
                    best_loss = val_J2J_loss
                    model_path = os.path.join(args.output_dir, 'best_model.pkl')
                    model.save(model_path)
                    log_message(f"Saved best model with J2J loss {best_loss:.6f} to {model_path}")
        
        if jt.rank == 0:
            # Save checkpoint regularly
            if (epoch + 1) % args.save_freq == 0:
                checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pkl')
                model.save(checkpoint_path)
                log_message(f"Saved checkpoint to {checkpoint_path}")
    
    if jt.rank == 0:
        # Save final model
        final_model_path = os.path.join(args.output_dir, 'final_model.pkl')
        model.save(final_model_path)
        log_message(f"Training completed. Saved final model to {final_model_path}")
    
    return model, best_loss

def main():
    """Parse arguments and start training"""
    parser = argparse.ArgumentParser(description='Train a keypoint codebook model')
    
    # Dataset parameters - removed sampling-related parameters
    parser.add_argument('--train_data_list', type=str, required=True,
                        help='Path to the training data list file')
    parser.add_argument('--val_data_list', type=str, default='',
                        help='Path to the validation data list file')
    parser.add_argument('--data_root', type=str, default='data',
                        help='Root directory for the data files')
    parser.add_argument('--rotation_range', type=float, default=90.0,
                        help='Rotation range for data augmentation')
    parser.add_argument('--scaling_range', type=float, nargs=2, default=(0.8, 1.2),
                        help='Scaling range for data augmentation')
    parser.add_argument('--aug_prob', type=float, default=0.5,
                        help='Probability of applying data augmentation')
    parser.add_argument('--drop_bad', action='store_true',
                        help='Drop bad samples based on predefined list')
    parser.add_argument('--pose_angle_range', type=float, default=30.0,
                        help='Range of pose angles for augmentation')
    parser.add_argument('--track_pose_aug', action='store_true',
                        help='Use tracking pose for augmentation')

    # Model parameters (updated for Transformer architecture)
    parser.add_argument('--model_type', type=str, default='keypoint_vqvae_transformer',
                        choices=['keypoint_vqvae_transformer', 'pose_aware_vqvae'],
                        help='Type of codebook model')
    parser.add_argument('--embedding_dim', type=int, default=512,
                        help='Dimension of the embedding vectors')
    parser.add_argument('--num_embeddings', type=int, default=1024,
                        help='Number of embedding vectors in codebook')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 256, 512],
                        help='Hidden dimensions for encoder/decoder (legacy, kept for compatibility)')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension for Transformer layers')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of Transformer layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads in Transformer')
    parser.add_argument('--commitment_cost', type=float, default=0.25,
                        help='Commitment cost for VQ loss')
    
    # Pretrained model
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='Path to pretrained model')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['sgd', 'adamw', 'adam'],
                        help='Optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['step', 'cosine'],
                        help='Learning rate scheduler type')
    parser.add_argument('--lr_step', type=int, default=20,
                        help='Step size for learning rate decay')
    parser.add_argument('--lr_decay', type=float, default=0.8,
                        help='Decay factor for learning rate')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                        help='Minimum learning rate for cosine scheduler')

    # Loss weights
    parser.add_argument('--vq_loss_weight', type=float, default=1.0,
                        help='Weight for vector quantization loss')
    parser.add_argument('--J2J_loss_weight', type=float, default=0.5,
                        help='Weight for J2J loss')
    parser.add_argument('--sym_loss_weight', type=float, default=0.05,
                        help='Weight for symmetry loss')
    parser.add_argument('--bone_length_symmetry_weight', type=float, default=0.5,
                        help='Weight for bone length symmetry loss')
    parser.add_argument('--topo_loss_weight', type=float, default=0.1,
                        help='Weight for topology loss')
    parser.add_argument('--rel_pos_loss_weight', type=float, default=0.1,
                        help='Weight for relative position loss')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='output/keypoint_codebook',
                        help='Directory to save output files')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save frequency')
    parser.add_argument('--val_freq', type=int, default=1,
                        help='Validation frequency')
    
    args = parser.parse_args()
    train(args)

def seed_all(seed):
    """Set seed for all random operations"""
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    seed_all(42)
    main()