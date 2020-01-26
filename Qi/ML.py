import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Enable full width output for numpy (https://stackoverflow.com/questions/43514106/python-terminal-output-width)
np.set_printoptions(suppress=True,linewidth=250,threshold=250)

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


# Read data
data = pd.read_feather('data.feather')
# Normalize curvature
data['Curvature'] = data['Curvature']/data['Curvature'].max()
data['Curvature'] = np.log(data['Curvature'])
# Split data
test_set, train_set = split_data(data, 0.2)

# Split the training and test data into labels (first column) and data
test_labels = test_set.iloc[:, 0].to_numpy()
test_data = test_set.iloc[:, 1:].to_numpy()
train_labels = train_set.iloc[:, 0].to_numpy()
train_data = train_set.iloc[:, 1:].to_numpy()
print(f'train_data: \n{train_data}')

# Build keras model
model = tf.keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=[9]),
    layers.Dense(1)
])

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mse', metrics=['mae', 'mse'])

# Fit Data
model.fit(train_data, train_labels, epochs=1000, batch_size=32, validation_split=0.2)
