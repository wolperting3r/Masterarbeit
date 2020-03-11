import pandas as pd

files = [
    'data_3x3_eqk_pos.feather',
    'data_5x5_eqk_pos.feather',
    'data_3x3_eqr_pos.feather',
    'data_5x5_eqr_pos.feather',
    'data_3x7_eqk_pos.feather',
    'data_7x3_eqk_pos.feather',
    'data_3x7_eqr_pos.feather',
    'data_7x3_eqr_pos.feather',
    'data_3x3_eqk_neg.feather',
    'data_5x5_eqk_neg.feather',
    'data_3x3_eqr_neg.feather',
    'data_5x5_eqr_neg.feather',
    'data_3x7_eqk_neg.feather',
    'data_7x3_eqk_neg.feather',
    'data_3x7_eqr_neg.feather',
    'data_7x3_eqr_neg.feather',
]

file_dfs = {}
for file in files:
    file_dfs[file] = pd.read_feather(file)
    print(file_dfs[file].shape)
    print(f'file_dfs[file].min():\n{file_dfs[file].min()}')
    print(f'file_dfs[file].max():\n{file_dfs[file].max()}')

