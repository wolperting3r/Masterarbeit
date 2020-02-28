import pandas as pd
import numpy as np
import os
import sys

from sklearn.pipeline import Pipeline
from src.d.transformators import TransformData, FindGradient, FindAngle, Rotate, FindKappa


# Enable full width output for numpy (https://stackoverflow.com/questions/43514106/python-terminal-output-width)
np.set_printoptions(suppress=True, linewidth=250, threshold=250)


def get_data(filename):
    ''' Import data from files '''
    parent_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    path = os.path.join(parent_path, 'data', 'datasets', filename)
    data = pd.read_feather(path)
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
    # Calculate how many entries the test data will have
    test_size = int(len(data)*ratio)
    # Get the test indices from the randomly generated indices
    test_indices = indices[:test_size]
    # Get the train indices from the randomly generated indices
    train_indices = indices[test_size:]
    # Return the data corresponding to the indices
    return data.iloc[test_indices], data.iloc[train_indices]


def transform_data(parameters, reshape=False):
    # Data file to load
    filename = 'data_' + \
            str(parameters['stencil_size'][0]) + 'x' + str(parameters['stencil_size'][1]) + '_' + \
            ('eqk' if parameters['equal_kappa'] else 'eqr') + \
            ('_neg' if parameters['negative'] else '_pos') + \
            '.feather'
    # Read data
    data = get_data(filename)
    # print(f'Imported data with shape {data.shape}')
    # Split data
    test_set, train_set = split_data(data, 0.2)

    '''
    data_pipeline = Pipeline([
        ('transform', TransformData(parameters=parameters, reshape=reshape)),
        ('findgradient', FindGradient(parameters=parameters)),
        ('findangle', FindAngle(parameters=parameters)),
        ('rotate', Rotate(parameters=parameters)),
    ])
    # '''
    data_pipeline = Pipeline([
        ('transform', TransformData(parameters=parameters, reshape=reshape)),
        ('findgradient', FindGradient(parameters=parameters)),
        ('findangle', FindAngle(parameters=parameters)),
    ])

    [test_labels, test_data, test_angle] = data_pipeline.fit_transform(test_set)
    # print(f'test_labels:\n{test_labels}')
    [train_labels, train_data, train_angle] = data_pipeline.fit_transform(train_set)
    # print(f'train_labels:\n{train_labels}')
    if parameters['angle']:
        test_data = np.hstack((test_data, test_angle))
        train_data = np.hstack((train_data, train_angle))

    '''
    ind = 6
    # print_data_orig = np.reshape(test_data, (test_data.shape[0], 5, 5, 1)).transpose((0, 1, 3, 2))[ind]
    print_data_grad = test_data.transpose((0, 1, 3, 2))[ind]
    # print(f'\nOriginal:\n{print_data_orig}')
    print(f'\nGedreht:\n{print_data_grad}')
    # '''

    return [[train_labels, train_data, train_angle], [test_labels, test_data, test_angle]]
