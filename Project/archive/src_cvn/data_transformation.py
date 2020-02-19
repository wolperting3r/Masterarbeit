# MACHINE LEARNING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm.keras import TqdmCallback
from sklearn.preprocessing import MinMaxScaler
from kerastuner.tuners import Hyperband

# Enable full width output for numpy (https://stackoverflow.com/questions/43514106/python-terminal-output-width)
np.set_printoptions(suppress=True, linewidth=250, threshold=250)


def get_data(filename):
    ''' Import data from files '''
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'datasets', filename)
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


def transform_data(filename, reshape=False, stencil=[]):
    # Read data
    data = get_data(filename)
    print(f'Imported data with shape {data.shape}')
    # Split data
    test_set, train_set = split_data(data, 0.2)
    # test_set = data[data['Curvature']>9]
    # train_set = data[data['Curvature']<=9]


    # Split the training and test data into labels (first column) and data
    test_labels = np.round(test_set.iloc[:, 0].to_numpy(), 3)
    test_data = np.round(test_set.iloc[:, 1:].to_numpy(), 3)
    train_labels = np.round([train_set.iloc[:, 0].to_numpy()], 3).T
    train_data = np.round(train_set.iloc[:, 1:].to_numpy(), 3)

    if reshape:
        if len(stencil)==0:
            # Infer stencil from filename:
            match = re.search(r'(\d)x(\d)', filename)
            st_sz = [int(match.group(1)), int(match.group(2))]
        # Reshape data
        test_data = np.reshape(test_data, (test_data.shape[0], st_sz[0], st_sz[1], 1))
        train_data = np.reshape(train_data, (train_data.shape[0], st_sz[0], st_sz[1], 1))
        # test_data = np.reshape(test_data, (test_data.shape[0], 1, st_sz[0], st_sz[1]))
        # train_data = np.reshape(train_data, (train_data.shape[0], 1, st_sz[0], st_sz[1]))

    return [[train_labels, train_data], [test_labels, test_data]]
