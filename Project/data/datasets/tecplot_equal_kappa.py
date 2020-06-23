import numpy as np
import os
import sys
import pandas as pd


def tecplot_equal_kappa(files, st_sz, filtering, filter):
    filtering = filtering[0]
    filter = filter[0]
    # Write output dataframe to feather file
    parent_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    # Create file name
    for f in files:
        filename = os.path.join(parent_path, 'data_CVOFLS_'+f+'_'+str(st_sz[0])+'x'+str(st_sz[1])+(f'_{filter}{filtering}' if filtering else '')+'.feather')
        print(f'File:\t{filename}')
        if not ('file' in locals()):
            file = pd.read_feather(filename)
            print(f'Shape:\t{file.shape}')
        else:
            temp = pd.read_feather(filename)
            print(f'Shape:\t{temp.shape}')
            file = file.append(temp, ignore_index=True)

    n_bins = 500
    data = file.values
    binwidth = (data[:, 0].max()-data[:, 0].min())/n_bins
    max_per_bin = 2e2

    output_data = np.empty((0, int(np.prod(st_sz)+1)))

    for i in range(n_bins):
        # Get lower and higher boundaries of bin
        bin_lower = data[:, 0].min()+i*binwidth
        bin_higher = data[:, 0].min()+(i+1)*binwidth
        # Get indices of data with curvature inside bin
        bin_indices = np.nonzero((data[:, 0] > bin_lower) & (data[:, 0] <= bin_higher))
        # Get data with that indices
        bin_data = data[bin_indices]
        # Bring indices into random order
        np.random.seed(2)
        indices = np.random.permutation(bin_data.shape[0])
        # Get data in random order
        bin_data = bin_data[indices]   
        if len(bin_indices[0]) >= max_per_bin:
            # If bin is larger then max_per_bin, get the first max_per_bin indices from random indices
            bin_data = bin_data[:int(min(max_per_bin, bin_data.shape[0]))]
            output_data = np.concatenate((output_data, bin_data))
        else:
            # If the bin is smaller than max_per_bin (but > 0), dublicate data (max 10 times, otherwise data will not be used)
            if len(bin_indices[0]) > 0:
                factor = int(np.ceil(max_per_bin/len(bin_indices[0])))
                if factor < 10:
                    for i in range(factor+1):
                        bin_data = np.concatenate((bin_data, bin_data), axis=0)
                    bin_data = bin_data[:int(min(max_per_bin, bin_data.shape[0]))]
                    output_data = np.concatenate((output_data, bin_data))

    out_filename = os.path.join(parent_path, 'data_CVOFLS_'+str(st_sz[0])+'x'+str(st_sz[1])+(f'_{filter}{filtering}' if filtering else '')+'_eqk.feather')

    # Convert output list to pandas dataframe
    output_df = pd.DataFrame(output_data)
    # Reformat column names as string and rename curvature column
    output_df.columns = output_df.columns.astype(str)
    output_df = output_df.rename(columns={'0':'Curvature'})
    print(f'File:\n{out_filename}')
    output_df.reset_index(drop=True).to_feather(out_filename)
    print(f'Generated {output_df.shape[0]} tuples with:\nStencil size:\t{st_sz}\n')

files = [
'128_00160028',
'32_0010033',
'64_0010033',
'128_0010033',
'128_0012003',
'256_0010033',
]

# st_szs = [[5, 5], [7, 7], [9, 9]]
st_szs = [[7, 7]]

filtering = [2]
filter = ['g']

for st_sz in st_szs:
    tecplot_equal_kappa(files, st_sz, filtering, filter)
