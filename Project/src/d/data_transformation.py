import pandas as pd
import numpy as np
import os
import sys
import time

from sklearn.pipeline import Pipeline
from src.d.transformators import TransformData, FindGradient, FindAngle, Rotate, CDS, HF, TwoLayers


# Enable full width output for numpy (https://stackoverflow.com/questions/43514106/python-terminal-output-width)
np.set_printoptions(suppress=True, linewidth=250, threshold=250)


def get_data(parameters):
    ''' Import data from files '''
    if parameters['data'] == 'all':
        # Data file to load
        filename_cir = 'data_' + \
            str(parameters['stencil_size'][0]) + 'x' + str(parameters['stencil_size'][1]) + '_' + \
            ('eqk' if parameters['equal_kappa'] else 'eqr') + \
            ('_neg' if parameters['negative'] else '_pos') + \
            '_sin' + \
            ('_smr' if parameters['smear'] else '_nsm') + \
            '.feather'
        filename_ell = 'data_' + \
            str(parameters['stencil_size'][0]) + 'x' + str(parameters['stencil_size'][1]) + '_' + \
            ('eqk' if parameters['equal_kappa'] else 'eqr') + \
            ('_neg' if parameters['negative'] else '_pos') + \
            '_ell' + \
            ('_smr' if parameters['smear'] else '_nsm') + \
            '.feather'
        parent_path = os.path.dirname(os.path.abspath(sys.argv[0]))
        print(f'filename_cir:\n{filename_cir}')
        print(f'filename_ell:\n{filename_ell}')
        path_sin = os.path.join(parent_path, 'data', 'datasets', filename_cir)
        path_ell = os.path.join(parent_path, 'data', 'datasets', filename_ell)
        data_sin = pd.read_feather(path_sin)
        data_ell = pd.read_feather(path_ell)
        data = pd.concat([data_sin, data_ell], ignore_index=True)
    else:
        if parameters['data'] == 'ellipse':
            geom_str = '_ell'
        elif parameters['data'] == 'sinus':
            geom_str = '_sin'
        elif parameters['data'] == 'circle':
            geom_str = '_cir'
        # Data file to load
        filename = 'data_' + \
            str(parameters['stencil_size'][0]) + 'x' + str(parameters['stencil_size'][1]) + '_' + \
            ('eqk' if parameters['equal_kappa'] else 'eqr') + \
            ('_neg' if parameters['negative'] else '_pos') + \
            geom_str + \
            ('_smr' if parameters['smear'] else '_nsm') + \
            '.feather'
        print(f'filename:\n{filename}')
        parent_path = os.path.dirname(os.path.abspath(sys.argv[0]))
        path = os.path.join(parent_path, 'data', 'datasets', filename)
        data = pd.read_feather(path)
    # print(f'Imported data with shape {data.shape}')
    return data.copy()


def split_data(data, ratio):
    '''
    Create randomized train set and test set based on the ratio
    ---
    Input:
        data[pd df]     data
        ratio[int]      test:train ratio
    Output:
        testdata[pd df]
        traindata[pd df]
    '''
    # Set seed
    np.random.seed(42)
    # Generate random indices
    indices = np.random.permutation(len(data))

    if isinstance(ratio, list):
        # [test_ratio, val_ratio, train_ratio] as list
        # Calculate how many entries the test data will have
        test_size = int(len(data)*ratio[1])
        val_size = int(len(data)*ratio[2])
    
        # Get the test indices from the randomly generated indices
        test_indices = indices[:test_size]
        # Get the validation indices from the randomly generated indices
        val_indices = indices[test_size:val_size+test_size]
        # Get the train indices from the randomly generated indices
        train_indices = indices[val_size+test_size:]
        # Return the data corresponding to the indices
        return data.iloc[test_indices], data.iloc[val_indices], data.iloc[train_indices]
    else:
        # test_ratio as float
        # Calculate how many entries the test data will have
        test_size = int(len(data)*ratio)
        # Get the test indices from the randomly generated indices
        test_indices = indices[:test_size]
        # Get the train indices from the randomly generated indices
        train_indices = indices[test_size:]
        # Return the data corresponding to the indices
        return data.iloc[test_indices], data.iloc[train_indices]


def process_data(dataset, parameters, reshape):
    '''
    # Calculate kappa with CDS
    # Create pipeline
    data_pipeline = Pipeline([
        ('transform', TransformData(parameters=parameters, reshape=reshape)),
        ('findkappacds', CDS(parameters=parameters)),
    ])
    # Execute pipeline
    [labels, features, angle] = data_pipeline.fit_transform(dataset)
    # '''
    if parameters['hf']:
        # Calculate kappa with HF
        # Create pipeline
        data_pipeline = Pipeline([
            ('transform', TransformData(parameters=parameters, reshape=reshape)),
            ('findgradient', FindGradient(parameters=parameters)),
            ('findkappahf', HF(parameters=parameters)),
        ])
        # Execute pipeline
        [labels, features, kappa] = data_pipeline.fit_transform(dataset)
    else:
        kappa = 0

    # '''
    # Pre-Processing
    if parameters['rotate']:
        # Create pipeline
        data_pipeline = Pipeline([
            ('transform', TransformData(parameters=parameters, reshape=reshape)),
            ('findgradient', FindGradient(parameters=parameters)),
            ('findangle', FindAngle(parameters=parameters)),
            ('rotate', Rotate(parameters=parameters)),  # Output: [labels, data, angle_matrix]
        ])
        # Execute pipeline
        [labels, features, angle] = data_pipeline.fit_transform(dataset)

    elif (parameters['angle'] & (not parameters['rotate'])):
        # Create pipeline
        data_pipeline = Pipeline([
            ('transform', TransformData(parameters=parameters, reshape=reshape)),
            ('findgradient', FindGradient(parameters=parameters)),
            ('findangle', FindAngle(parameters=parameters)),  # Output: [labels, data, angle_matrix]
        ])
        # Execute pipeline
        [labels, features, angle] = data_pipeline.fit_transform(dataset)

    else:
        # Create pipeline
        data_pipeline = Pipeline([
            ('transform', TransformData(parameters=parameters, reshape=reshape)),  # Output: [labels, data]
        ])
        # Execute pipeline
        [labels, features] = data_pipeline.fit_transform(dataset)
        # Dummy angle
        angle = np.zeros(features.shape)
    # '''

    if parameters['angle']:
        # Stack features and angles
        features = np.hstack((features, angle))

    if parameters['hf_correction']:
        if parameters['network'] == 'cvn':
            # Write kappa into second layer on 3rd axis
            data_pipeline = Pipeline([
                ('twolayers', TwoLayers(parameters=parameters)),
            ])
            [labels, features] = data_pipeline.fit_transform([labels, features, kappa])
        else:
            # Stack features and height function
            features = np.concatenate((features, kappa), axis=1)

    return [labels, features, angle, kappa]



def transform_data(parameters, reshape=False, plot=False):
    time0 = time.time()
    # Read data
    data = get_data(parameters)

    # Split data
    test_set, val_set, train_set = split_data(data, [0.15, 0.15, 0.7])

    # Pre-Processing
    test_labels, test_data, test_angle, test_kappa = process_data(test_set, parameters, reshape)
    if not plot:
        train_labels, train_data, train_angle, train_kappa = process_data(train_set, parameters, reshape)
        val_labels, val_data, val_angle, val_kappa = process_data(val_set, parameters, reshape)
    else:
        train_labels, train_data, train_angle, train_kappa = 0, 0, 0, 0
        val_labels, val_data, val_angle, val_kappa = 0, 0, 0, 0


    filestring = parameters['filename']
    print(f'Time needed for pre-processing of {filestring}:\t{np.round(time.time()-time0,3)}s')

    return [
        [train_labels, train_data, train_angle, train_kappa],
        [test_labels, test_data, test_angle, test_kappa],
        [val_labels, val_data, val_angle, val_kappa],
    ]

    '''
    # Test
    ind = 6
    # print_data_orig = np.reshape(test_data, (test_data.shape[0], 5, 5, 1)).transpose((0, 1, 3, 2))[ind]
    print_data_grad = test_data.transpose((0, 1, 3, 2))[ind]
    # print(f'\nOriginal:\n{print_data_orig}')
    print(f'\nGedreht:\n{print_data_grad}')
    # '''

