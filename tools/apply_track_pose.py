import numpy as np
from dataset.asset import Asset
matrix_basis = np.load('dataB/track/0.npz')['matrix_basis'][6]
asset = Asset.load("dataB/train/vroid/0.npz")
# for matrix in matrix_basis:
#     asset.apply_matrix_basis(matrix)
asset.apply_matrix_basis(matrix_basis)
asset.export_mesh('res6.obj') # 导出查看