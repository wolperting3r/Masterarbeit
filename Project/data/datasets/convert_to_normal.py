import pandas as pd

data = pd.read_feather('data_5x5_eqk_neg_ell_flat_e.feather')
data = data.rename(columns={'Curvature':'pt_x', '1': 'pt_y', '2': 'e', '3': 'r', '4': 'curvature'})
data = data.drop(['pt_x', 'pt_y', 'e', 'r'], axis=1)
data.columns = list(range(data.shape[1]))
data.columns = data.columns.astype(str)
data = data.rename(columns={'0': 'Curvature'})
data.to_feather('data_5x5_eqk_neg_ell_flat_e_n.feather')
print(f'data.columns:\n{data.columns}')
