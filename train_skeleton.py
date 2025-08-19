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
from dataset.sampler import SamplerMix
from dataset.exporter import Exporter
from models.skeleton import create_model

from models.metrics import (
    J2J, bone_length_symmetry_loss, symmetry_loss, chamfer_distance, 
    topology_loss, relative_position_loss, mesh_interior_loss_advanced,
    mesh_interior_loss_fast_normals, skeleton_mesh_consistency_loss, mesh_interior_loss_hierarchical
)

import wandb
os.environ["WANDB_MODE"] = "offline"

# if hard
dataset_with_hand = os.environ.get('HARD', '0').lower() in ['1', 'true', 'yes', 'on']
print(f'Joints: {52 if dataset_with_hand else 22}')
# import pdb;pdb.set_trace()
# Set Jittor flags
jt.flags.use_cuda = 1

def train(args):
    """
    Main training function
    
    Args:
        args: Command line arguments
    """
    if jt.rank == 0:
        # Create output directory if it doesn't exist
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        # Set up logging
        log_file = os.path.join(args.output_dir, 'training_log.txt')
        vital_files = ['train_skeleton.py', 'models/skeleton.py', 'dataset/dataset.py']
        for file in vital_files:
            shutil.copy(file, args.output_dir)
        # writer = SummaryWriter(logdir=args.output_dir)
        wandb.init(project="jittor-skeleton-training", config=args, name=f"skeleton_{args.model_name}", dir=args.output_dir)
        wandb.run.name = f"skeleton_{time.strftime('%Y%m%d_%H%M%S')}"

        def log_message(message):
            """Helper function to log messages to file and print to console"""
            with open(log_file, 'a') as f:
                f.write(f"{message}\n")
            print(message)
        
        # Log training parameters
        log_message(f"Starting training with parameters: {args}")
    
    # Create model
    num_skeletons = 52 if dataset_with_hand else 22
    model = create_model(
        model_name=args.model_name,
        model_type=args.model_type,
        output_channels=num_skeletons * 3,  # Each joint has 3 coordinates (x, y, z)
        with_normals=args.wnormals,
        num_tokens=args.num_tokens,
        feat_dim=args.feat_dim,
        encoder_layers=args.encoder_layers
    )
    
    sampler = SamplerMix(num_samples=args.num_samples, vertex_samples=args.vertex_samples)

    # Load pre-trained model if specified
    if args.pretrained_model:
        if jt.rank==0: 
            log_message(f"Loading pretrained model from {args.pretrained_model}")

        if dataset_with_hand:
            model.load_with_skeleton_transfer(args.pretrained_model, num_pretrained_skeletons=22)
        else:
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
    
    # create lr scheduler
    if args.lr_scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_decay)
    elif args.lr_scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)

    # Create loss function
    criterion = nn.MSELoss()
    
    # Create dataloaders
    train_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.train_data_list,
        train=True,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=sampler,
        transform=transform,
        rotation_range=args.rotation_range,
        scaling_range=args.scaling_range,
        aug_prob=args.aug_prob,
        drop_bad=args.drop_bad,
        pose_angle_range=args.pose_angle_range,
        track_pose_aug=args.track_pose_aug,
        hand=dataset_with_hand,
    )
    
    if args.val_data_list:
        val_loader = get_dataloader(
            data_root=args.data_root,
            data_list=args.val_data_list,
            train=False,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            transform=transform,
            pose_angle_range=0, 
            track_pose_aug=args.track_pose_aug,
            hand=dataset_with_hand,
        )
    else:
        val_loader = None
    
    # Training loop
    best_loss = 99999999
    for epoch in range(args.epochs):
        # Training phase
        model.train()

        if jt.rank==0:
            train_loss = 0.0
            loss_dict = {
                'mse': 0.0,
                'J2J': 0.0,
                'symmetry': 0.0,
                'bone_length_symmetry': 0.0,
                'topology': 0.0,
                'relative_position': 0.0,
                'mesh_interior': 0.0
            }
            start_time = time.time()
        
        for batch_idx, data in enumerate(train_loader):
            # Get data and labels
            vertices, joints, normals = data['vertices'], data['joints'], data['normals']

            if args.model_name == 'sal':
                outputs = model(vertices, normals)
            else:
                outputs = model(vertices)

            outputs = outputs.reshape(outputs.shape[0], -1, 3)  # [B, J, 3]
            # joints = joints.reshape(outputs.shape[0], -1)

            mse_loss = criterion(outputs, joints)
            J2J_loss =  chamfer_distance(outputs, joints)
            sym_loss = symmetry_loss(outputs, joints)
            sym_bone_loss = bone_length_symmetry_loss(outputs)
            topo_loss = topology_loss(outputs, joints)
            rel_pos_loss = relative_position_loss(outputs, joints)
            
            # 添加mesh内部约束损失
            # 优先使用基于法向量的方法
            if 'normals' in data and hasattr(args, 'use_normals_interior') and args.use_normals_interior:
                out_skeleton = [9, 13, 17, 21]
                inmask = jt.ones(outputs.shape[1], dtype=jt.bool)
                if args.terminal_interior_loss:
                    # 如果开启了终端关节的内部约束损失，则不计算手脚的内部约束损失
                    inmask[out_skeleton] = False    
                mesh_interior_loss = mesh_interior_loss_hierarchical(outputs[:, inmask], vertices, normals, 
                                                                       margin=args.interior_margin)
            else:
                mesh_interior_loss = mesh_interior_loss_advanced(outputs[:, inmask], vertices, 
                                                                  k_neighbors=args.interior_k_neighbors, 
                                                                  margin=args.interior_margin)

            loss = mse_loss + sym_loss * args.sym_loss_weight \
                 + sym_bone_loss * args.bone_length_symmetry_weight \
                 + J2J_loss * args.J2J_loss_weight \
                 + topo_loss * args.topo_loss_weight \
                 + rel_pos_loss * args.rel_pos_loss_weight \
                 + mesh_interior_loss * args.mesh_interior_weight


            # Backward pass and optimize
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            scheduler.step()  # Step the scheduler

            if jt.in_mpi:
                loss = loss.mpi_all_reduce()
                mse_loss = mse_loss.mpi_all_reduce()
                J2J_loss = J2J_loss.mpi_all_reduce()
                sym_loss = sym_loss.mpi_all_reduce()
                sym_bone_loss = sym_bone_loss.mpi_all_reduce()
                topo_loss = topo_loss.mpi_all_reduce()
                rel_pos_loss = rel_pos_loss.mpi_all_reduce()
                mesh_interior_loss = mesh_interior_loss.mpi_all_reduce()

            if jt.rank==0:
                # Calculate statistics
                train_loss += loss.item()

                # Update loss statistics
                loss_dict['mse'] += mse_loss.item()
                loss_dict['J2J'] += J2J_loss.item()
                loss_dict['symmetry'] += sym_loss.item()
                loss_dict['bone_length_symmetry'] += sym_bone_loss.item()
                loss_dict['topology'] += topo_loss.item()
                loss_dict['relative_position'] += rel_pos_loss.item()
                loss_dict['mesh_interior'] += mesh_interior_loss.item()

                
                # Print progress
                if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(train_loader):
                    log_message(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Train Loss: {train_loss/(batch_idx+1):.4f} "
                            f"MSE Loss: {loss_dict['mse']/(batch_idx+1):.4f} J2J Loss: {loss_dict['J2J']/(batch_idx+1):.6f} Sym Loss: {loss_dict['symmetry']/(batch_idx+1):.4f} "
                            f"SymBone Loss: {loss_dict['bone_length_symmetry']/(batch_idx+1):.4f} Topo Loss: {loss_dict['topology']/(batch_idx+1):.4f} "
                            f"RelPos Loss: {loss_dict['relative_position']/(batch_idx+1):.4f} Interior Loss: {loss_dict['mesh_interior']/(batch_idx+1):.4f}")
                    global_step = epoch * len(train_loader) + batch_idx
                    wandb.log({
                        "skeleton epoch": epoch + 1,
                        "skeleton total_loss": train_loss,
                        "skeleton pos_loss": loss_dict['mse'] / (batch_idx + 1),
                        "skeleton topo_loss": loss_dict['topology'] / (batch_idx + 1),
                        "skeleton rel_loss": loss_dict['relative_position'] / (batch_idx + 1),
                        "skeleton sym_loss": loss_dict['symmetry'] / (batch_idx + 1),
                        "skeleton interior_loss": loss_dict['mesh_interior'] / (batch_idx + 1),
                        "skeleton learning_rate": optimizer.lr if hasattr(optimizer, 'lr') else args.learning_rate
                    }, step=global_step)
        
        if jt.rank==0:
            # Calculate epoch statistics
            train_loss /= len(train_loader)
            epoch_time = time.time() - start_time
            
            log_message(f"Epoch [{epoch+1}/{args.epochs}] "
                    f"Train Loss: {train_loss:.4f} "
                    f"Time: {epoch_time:.2f}s "
                    f"LR: {optimizer.lr:.6f}"
                    f" MSE Loss: {loss_dict['mse']/len(train_loader):.4f} "
                    f"J2J Loss: {loss_dict['J2J']/len(train_loader):.6f} "
                    f"Sym Loss: {loss_dict['symmetry']/len(train_loader):.4f} "
                    f"SymBone Loss: {loss_dict['bone_length_symmetry']/len(train_loader):.4f} "
                    f"Topo Loss: {loss_dict['topology']/len(train_loader):.4f} "
                    f"RelPos Loss: {loss_dict['relative_position']/len(train_loader):.4f} "
                    f"Interior Loss: {loss_dict['mesh_interior']/len(train_loader):.4f}")
        

        
        # Validation phase
        if val_loader is not None and (epoch + 1) % args.val_freq == 0:
            model.eval()
            if jt.rank ==0 :
                val_loss = 0.0
                J2J_loss = 0.0
                show_id = np.random.randint(0, len(val_loader))

            for batch_idx, data in enumerate(val_loader):
                # Get data and labels
                vertices, joints, normals = data['vertices'], data['joints'], data['normals']

                # Forward pass
                if args.model_name == 'sal':
                    outputs = model(vertices, normals)
                else:
                    outputs = model(vertices)
                outputs = outputs.reshape(outputs.shape[0], -1, 3)  # [B, J, 3]
                loss = criterion(outputs, joints)

                if jt.in_mpi:
                    loss = loss.mpi_all_reduce()

                if jt.rank==0:
                    # export render results
                    if batch_idx == show_id:
                        exporter = Exporter()
                        # export every joint's corresponding skinning
                        from dataset.format import parents
                        exporter._render_skeleton(path=f"{args.output_dir}/tmp/skeleton/epoch_{epoch}/skeleton_ref.png", joints=joints[0].numpy().reshape(-1, 3), parents=parents)
                        exporter._render_skeleton(path=f"{args.output_dir}/tmp/skeleton/epoch_{epoch}/skeleton_pred.png", joints=outputs[0].numpy().reshape(-1, 3), parents=parents)
                        exporter._render_pc(path=f"{args.output_dir}/tmp/skeleton/epoch_{epoch}/vertices.png", vertices=vertices[0].permute(1, 0).numpy())

                    val_loss += loss.item()
                    for i in range(outputs.shape[0]):
                        J2J_loss += J2J(outputs[i].reshape(-1, 3), joints[i].reshape(-1, 3)).item() / outputs.shape[0]
            
            if jt.rank==0:
                # Calculate validation statistics
                val_loss /= len(val_loader)
                J2J_loss /= len(val_loader)
                
                log_message(f"Validation Loss: {val_loss:.4f} J2J Loss: {J2J_loss:.4f}")
                # writer.add_scalar('val/loss', val_loss, epoch)
                # writer.add_scalar('val/J2J_loss', J2J_loss, epoch)

                # Save best model
                if J2J_loss < best_loss:
                    best_loss = J2J_loss
                    model_path = os.path.join(args.output_dir, 'best_model.pkl')
                    model.save(model_path)
                    log_message(f"Saved best model with loss {best_loss:.4f} to {model_path}")
        
        if jt.rank==0:
            # Save checkpoint
            if (epoch + 1) % args.save_freq == 0:
                checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pkl')
                model.save(checkpoint_path)
                log_message(f"Saved checkpoint to {checkpoint_path}")
    
    if jt.rank==0:
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
    parser.add_argument('--rotation_range', type=float, default=0.0,
                        help='Rotation range for data augmentation')
    parser.add_argument('--scaling_range', type=float, nargs=2, default=(1.0, 1.0),
                        help='Scaling range for data augmentation')
    parser.add_argument('--aug_prob', type=float, default=0.0,
                        help='Probability of applying data augmentation (rotation/scaling)')
    parser.add_argument('--drop_bad', action='store_true',
                        help='Drop bad samples based on predefined list')
    parser.add_argument('--pose_angle_range', type=float, default=0.0,
                        help='Range of pose angles for augmentation')
    parser.add_argument('--num_skeletons', type=int, default=52,
                        help='Number of skeletons to predict')

    # Model parameters
    parser.add_argument('--model_name', type=str, default='pct',
                        choices=['pct', 'pct2', 'sal', 'custom_pct', 'skeleton'],
                        help='Model architecture to use')
    parser.add_argument('--model_type', type=str, default='standard',
                        choices=['standard', 'enhanced'],
                        help='Model type for skeleton model')
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='Path to pretrained model')
    parser.add_argument('--wnormals', action='store_true',
                        help='Use normals in the model input')
    parser.add_argument('--num_tokens', type=int, default=512,
                        help='Number of tokens for SAL model')
    parser.add_argument('--feat_dim', type=int, default=512,
                        help='Feature dimension for the model')
    parser.add_argument('--encoder_layers', type=int, default=8,
                        help='Number of layers in the encoder')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['sgd', 'adamw'],
                        help='Optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=0.00001,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    parser.add_argument('--lr_step', type=int, default=20,
                        help='Step size for learning rate decay')
    parser.add_argument('--lr_decay', type=float, default=0.8,
                        help='Decay factor for learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='step',
                        choices=['step', 'cosine'],
                        help='Learning rate scheduler type')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                        help='Minimum learning rate for cosine scheduler')
    parser.add_argument('--sym_loss_weight', type=float, default=0.01,
                        help='Weight for symmetry loss')
    parser.add_argument('--bone_length_symmetry_weight', type=float, default=0.1,
                        help='Weight for bone length symmetry loss')
    parser.add_argument('--J2J_loss_weight', type=float, default=0.1,
                        help='Weight for J2J loss')
    parser.add_argument('--topo_loss_weight', type=float, default=0.1,
                        help='Weight for topology loss')
    parser.add_argument('--rel_pos_loss_weight', type=float, default=0.1,
                        help='Weight for relative position loss')
    parser.add_argument('--mesh_interior_weight', type=float, default=0.5,
                        help='Weight for mesh interior constraint loss')
    parser.add_argument('--interior_margin', type=float, default=0.01,
                        help='Safety margin for interior constraint')
    parser.add_argument('--interior_k_neighbors', type=int, default=50,
                        help='Number of neighbors for interior loss calculation')
    parser.add_argument('--use_normals_interior', action='store_true',
                        help='Use normals-based interior constraint (faster and more accurate)')
    parser.add_argument('--terminal_interior_loss', action='store_true',
                        help='Use terminal joints for interior loss (faster and more accurate)')
    parser.add_argument('--track_pose_aug', action='store_true',
                        help='Use tracking pose for augmentation')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='output/skeleton',
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