import jittor as jt
import numpy as np
import os
import argparse

from dataset.dataset import get_dataloader, transform
from dataset.sampler import SamplerMix
from dataset.exporter import Exporter
from models.skeleton import create_model

from tqdm import tqdm

import random

# Set Jittor flags
jt.flags.use_cuda = 1

def predict(args):
    # Create model
    model = create_model(
        model_name=args.model_name,
        model_type=args.model_type,
        output_channels=args.num_skeletons * 3,
        with_normals=args.wnormals,
        num_tokens=args.num_tokens,
        feat_dim=args.feat_dim,
        encoder_layers=args.encoder_layers
    )

    sampler = SamplerMix(num_samples=args.num_samples, vertex_samples=args.vertex_samples)
    
    # Load pre-trained model if specified
    if args.pretrained_model:
        model.load(args.pretrained_model)
    model.eval()
    
    predict_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.predict_data_list,
        train=False,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        transform=transform,
    )
    predict_output_dir = args.predict_output_dir
    print("start predicting...")
    exporter = Exporter()
    for batch_idx, data in tqdm(enumerate(predict_loader)):
        vertices, cls, id, normals = data['vertices'], data['cls'], data['id'], data['normals']
        # Reshape input if needed
        B = vertices.shape[0]
        if args.wnormals:
            outputs = model(vertices, normals)
        else:
            outputs = model(vertices)
        outputs = outputs.reshape(B, -1, 3)
        print(outputs.shape)
        for i in range(len(cls)):
            path = os.path.join(predict_output_dir, cls[i], str(id[i].item()))
            os.makedirs(path, exist_ok=True)
            np.save(os.path.join(path, "predict_skeleton"), outputs[i])
    print("finished")

def main():
    """Parse arguments and start training"""
    parser = argparse.ArgumentParser(description='Train a point cloud model')
    
    # Dataset parameters
    parser.add_argument('--data_root', type=str, default='data',
                        help='Root directory for the data files')
    parser.add_argument('--predict_data_list', type=str, required=True,
                        help='Path to the prediction data list file')
    parser.add_argument('--num_samples', type=int, default=1024,
                        help='Number of samples to use for each point cloud')
    parser.add_argument('--vertex_samples', type=int, default=512,
                        help='Number of vertex samples to use for each point cloud')
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
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--wnormals', action='store_true',
                        help='Use normals in the model input')
    parser.add_argument('--num_tokens', type=int, default=512,
                        help='Number of tokens for SAL model')
    parser.add_argument('--feat_dim', type=int, default=512,
                        help='Feature dimension for the model')
    parser.add_argument('--encoder_layers', type=int, default=8,
                        help='Number of heads in the encoder')
    
    # Predict parameters
    parser.add_argument('--predict_output_dir', type=str,
                        help='Path to store prediction results')
    
    args = parser.parse_args()
    
    predict(args)

def seed_all(seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    seed_all(123)
    main()