# MACHINE LEARNING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm.keras import TqdmCallback
from sklearn.preprocessing import MinMaxScaler

# Enable full width output for numpy (https://stackoverflow.com/questions/43514106/python-terminal-output-width)
np.set_printoptions(suppress=True, linewidth=250, threshold=250)

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


def get_data(source):
    '''
    Import data from files
    ---
    Input:
        source[str]   'self' or 'paper'
    Output:
        data[pd df] data
    '''
    if source == 'self':
        data = pd.read_feather('data.feather')
        return data.copy()
    elif source == 'paper':
        csv_data = pd.read_csv('datafra.csv', names=['1', '2', '3', '4', '5', '6', '7', '8', '9']).reset_index()
        csv_labels = pd.read_csv('datacur.csv', names=['Curvature']).reset_index()
        data = csv_labels.merge(csv_data, on='index').drop(columns=['index'], axis=1)
        return data.copy()
    else:
        print('invalid from')


def build_model():
    # Build keras model
    model = tf.keras.Sequential([
        layers.Dense(70, activation='tanh', kernel_initializer='he_normal', input_shape=(27,)),
        layers.Dense(50, activation='tanh'),
        layers.Dense(20, activation='tanh'),
        layers.Dense(1, activation='linear')
    ])

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='mse', metrics=['mae', 'mse'])
    return model


def train_model(model, train_data, train_labels, regenerate=True):
    # Build tensorflow dataset
    dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(16)
    if regenerate:
        # Train Model
        # model.fit(train_data, train_labels, verbose=0, callbacks=[TqdmCallback(verbose=1)], epochs=1000, batch_size=8192, validation_split=0.2)
        early_stopping_callback = keras.callbacks.EarlyStopping(monitor='loss', min_delta=10e-8, patience=50, verbose=0, mode='auto', baseline=None)
        model.fit(dataset, verbose=0, callbacks=[TqdmCallback(verbose=1), early_stopping_callback], epochs=10000)

        # Save model
        model.save('model.h5')
    else:
        model = tf.keras.models.load_model('model.h5')
        print(model.summary())

    return model


def validate_model(model, train_data, train_labels):
    # Validate model
    train_predictions = model.predict(train_data, batch_size=64).flatten()

    plt.axes(aspect='equal')
    plt.scatter(train_labels, train_predictions, alpha=0.1)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    lims = [-0.25, 0.25]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()


if __name__ == '__main__':
    # Read data
    data = get_data('self')

    # data = data.iloc[30:40,:]
    # Normalize curvature
    # data['Curvature'] = data['Curvature']/data['Curvature'].max()
    # data['Curvature'] = np.log(data['Curvature'])

    # Split data
    test_set, train_set = split_data(data, 0.2)

    # Split the training and test data into labels (first column) and data
    test_labels = np.round(test_set.iloc[:, 0].to_numpy(), 3)
    test_data = np.round(test_set.iloc[:, 1:].to_numpy(), 3)
    train_labels = np.round([train_set.iloc[:, 0].to_numpy()], 3).T
    train_data = np.round(train_set.iloc[:, 1:].to_numpy(), 3)

    # scaler = MinMaxScaler()
    # train_labels = scaler.fit_transform(train_labels)
    # print(f'train_data: \n{train_data}')
    # print(f'train_labels: \n{train_labels}')
    # '''
    model = build_model()

    model = train_model(model, train_data, train_labels)

    validate_model(model, train_data, train_labels)
    # '''
