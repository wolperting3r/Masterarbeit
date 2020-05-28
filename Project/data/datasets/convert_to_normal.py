import pandas as pd

'''
data = pd.read_feather('data_5x5_eqk_neg_ell_smr.feather')
data = data.rename(columns={'Curvature':'pt_x', '1': 'pt_y', '2': 'e', '3': 'r', '4': 'curvature'})
data = data.drop(['pt_x', 'pt_y', 'e', 'r'], axis=1)
data.columns = list(range(data.shape[1]))
data.columns = data.columns.astype(str)
data = data.rename(columns={'0': 'Curvature'})
data.to_feather('data_5x5_eqk_neg_ell_smr2.feather')
print(f'data.columns:\n{data.columns}')

data = pd.read_feather('data_3x3_eqk_neg_ell_smr.feather')
data = data.rename(columns={'Curvature':'pt_x', '1': 'pt_y', '2': 'e', '3': 'r', '4': 'curvature'})
data = data.drop(['pt_x', 'pt_y', 'e', 'r'], axis=1)
data.columns = list(range(data.shape[1]))
data.columns = data.columns.astype(str)
data = data.rename(columns={'0': 'Curvature'})
data.to_feather('data_3x3_eqk_neg_ell_smr2.feather')

data = pd.read_feather('data_7x7_eqk_neg_ell_smr.feather')
data = data.rename(columns={'Curvature':'pt_x', '1': 'pt_y', '2': 'e', '3': 'r', '4': 'curvature'})
data = data.drop(['pt_x', 'pt_y', 'e', 'r'], axis=1)
data.columns = list(range(data.shape[1]))
data.columns = data.columns.astype(str)
data = data.rename(columns={'0': 'Curvature'})
data.to_feather('data_7x7_eqk_neg_ell_smr2.feather')
'''
files = [
    'data_7x7_eqk_neg_sin_smr',
    'data_7x7_eqk_neg_sin_nsm',
    'data_5x5_eqk_neg_sin_smr',
    'data_5x5_eqk_neg_sin_nsm',
]

for file in files:
    data = pd.read_feather(file + '.feather')
    print(f'data.iloc[:10,0]:\n{data.iloc[:10,0]}')
    data['Curvature'] = -data['Curvature']
    print(f'data.iloc[:10,0]:\n{data.iloc[:10,0]}')
    data.to_feather(file + '_copy.feather')
    print(f'data.columns:\n{data.columns}')
