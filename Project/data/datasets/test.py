import pandas as pd
import numpy as np

files = [
    'data_7x7_eqk_neg_sin_smr.feather',
    'data_7x7_eqk_neg_ell_smr.feather',
    'data_7x7_eqk_neg_cir_smr.feather',
]

file_dfs = {}
for file in files:
    file_dfs[file] = pd.read_feather(file)
    print(file_dfs[file].shape)
    # print(f'file_dfs[file].min():\n{file_dfs[file].min()}')
    # print(f'file_dfs[file].max():\n{file_dfs[file].max()}')
    curvature = file_dfs[file]['Curvature']
    print(f'curvature.max():\t\t{curvature.max()}')
    print(f'curvature.min():\t\t{curvature.min()}')
    print(f'np.abs(curvature).min():\t{np.abs(curvature).min()}')


