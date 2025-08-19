# import jittor as jt
import numpy as np
import sys
sys.path.append('.')
from dataset.asset import Asset
import os

# file = 'train/mixamo/819.npz'
with open('dataB/val_list.txt', 'r') as f:
    files = f.read().splitlines()

for file in files:
    item = file.split('.')[0]
    file_path = os.path.join('dataB', file)
    asset = Asset.load(path=file_path)
    folder = f'render/fbx_b/{item}'
    folder = folder[:folder.rfind('/')]
    os.makedirs(folder, exist_ok=True)
    asset.export_fbx(f'render/fbx_b/{item}.fbx')

