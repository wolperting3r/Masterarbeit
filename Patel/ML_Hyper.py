# MACHINE LEARNING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm.keras import TqdmCallback
from sklearn.preprocessing import MinMaxScaler
from kerastuner.tuners import Hyperband

# Enable full width output for numpy (https://stackoverflow.com/questions/43514106/python-terminal-output-width)
np.set_printoptions(suppress=True, linewidth=250, threshold=250)


def relative_mse(y_true, y_pred):
    return tf.reduce_mean(tf.math.squared_difference(y_pred,y_true))/y_true


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
    else:
        print('invalid from')


def build_model(hp):
    # Build keras model
    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=(9,)))
    for i in range(hp.Int('num_layers', 1, 1)):
        model.add(layers.Dense(units=hp.Choice('units_' + str(i), [100, 500]),
                               activation=hp.Choice('activation', ['relu', 'tanh'])))
    model.add(layers.Dense(units=1, activation='linear'))

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-4, 5e-5, 1e-5])),
                  loss='mse',
                  metrics=['mae', 'mse', relative_mse])
    return model


def train_model(model, train_data, train_labels, regenerate=False):
    # Build tensorflow dataset
    dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(128)

    if regenerate:
        # Train Model
        early_stopping_callback = keras.callbacks.EarlyStopping(monitor='loss',
                                                                min_delta=10e-8,
                                                                patience=50,
                                                                verbose=0,
                                                                mode='auto',
                                                                baseline=None)
        '''
        model.fit(dataset,
                  shuffle=True,
                  epochs=500,  # war 10000
                  verbose=0,
                  callbacks=[TqdmCallback(verbose=1),
                             early_stopping_callback])
        '''
        tuner.search(dataset,
                     shuffle=True,
                     epochs=500,  # war 10000
                     verbose=0,
                     callbacks=[TqdmCallback(verbose=1),
                                early_stopping_callback])
        model = tuner.get_best_models(num_models=1)
        print(tuner.results_summary())
        # Save model
        model.save('model.h5')
    else:
        model = tf.keras.models.load_model('model.h5')
        print(model.summary())

    return model


def validate_model(model, test_data, test_labels):
    # Validate model
    test_predictions = model.predict(test_data, batch_size=128).flatten()
    # Print MSE and MAE
    print(model.evaluate(test_data, test_labels, verbose=2))

    plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions, alpha=0.1)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    lims = [min(test_labels), max(test_labels)]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()


if __name__ == '__main__':
    # Read data
    data = get_data('self')
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

    # scaler = MinMaxScaler()
    # train_labels = scaler.fit_transform(train_labels)
    # print(f'train_data: \n{train_data}')
    # print(f'train_labels: \n{train_labels}')
    # '''
    # model = build_model()
    # Build tuner
    tuner = Hyperband(
        build_model,
        objective='mse',
        max_epochs=100,
        executions_per_trial=5,
        directory='tuning',
        project_name='tuning')
    print(tuner.search_space_summary())

    # Build tensorflow dataset
    dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(128)

    # Train Model
    tuner.search(dataset,
                 shuffle=True,
                 epochs=500,  # war 10000
                 verbose=0)
    model = tuner.get_best_models(num_models=1)
    print(tuner.results_summary())
    # Save model
    # model.save('model.h5')

    validate_model(model, test_data, test_labels)
    # validate_model(model, train_data, train_labels)
    # '''
