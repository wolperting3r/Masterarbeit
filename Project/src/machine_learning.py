# MACHINE LEARNING
import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm.keras import TqdmCallback

import io
from contextlib import redirect_stdout

from .data_transformation import transform_data

# Enable full width output for numpy (https://stackoverflow.com/questions/43514106/python-terminal-output-width)
np.set_printoptions(suppress=True, linewidth=250, threshold=250)


def relative_mse(y_true, y_pred):
    # https://stackoverflow.com/questions/51700351/valueerror-unknown-metric-function-when-using-custom-metric-in-keras 
    return tf.reduce_mean(tf.math.squared_difference(y_pred, y_true))/y_true


def param_filename(parameters):
    # Generate filename string
    filename_string = ''
    for key, value in parameters.items():
        if key == 'layers':
            filename_string = filename_string + '_' + '-'.join(str(e) for e in value)
        elif key == 'stencil_size':
            filename_string = filename_string + '_' + str(value[0]) + 'x' + str(value[1])
        elif key == 'equal_kappa':
            filename_string = filename_string + '_' + ('eqk' if value else 'eqr')
        else:
            filename_string = filename_string + '_' + str(value)
    return filename_string


def build_model(parameters, shape):
    # Build keras model
    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=(shape,)))
    for l in parameters['layers']:
        model.add(layers.Dense(l, activation=parameters['activation']))
    model.add(layers.Dense(1, activation='linear'))

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(parameters['learning_rate']),
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model


def train_model(model, train_data, train_labels, parameters, silent, regenerate=True):
    # Build tensorflow dataset
    # dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(128)
    dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(parameters['batch_size'])
    if regenerate:
        # Train Model
        early_stopping_callback = keras.callbacks.EarlyStopping(monitor='loss',
                                                                min_delta=10e-8,
                                                                patience=50,
                                                                verbose=0,
                                                                mode='auto',
                                                                baseline=None)
        model.fit(dataset,
                  shuffle=True,
                  epochs=parameters['epochs'],  # war 10000
                  verbose=0,
                  callbacks=[TqdmCallback(verbose=(0 if silent else 1)),
                             early_stopping_callback])

        # Save model
        path = os.path.dirname(os.path.abspath(__file__))
        param_str = parameters['filename']
        file_name = os.path.join(path, '..', 'models', 'model' + param_str + '.h5')
        model.save(file_name)
    else:
        path = os.path.dirname(os.path.abspath(__file__))
        param_str = parameters['filename']
        file_name = os.path.join(path, '..', 'models', 'model' + param_str + '.h5')
        model = tf.keras.models.load_model(file_name)
        print(model.summary())

    return model


def validate_model(model, train_data, train_labels, test_data, test_labels, parameters, plot=True):
    if not plot:
        # Print MSE and MAE
        path = os.path.dirname(os.path.abspath(__file__))
        param_str = parameters['filename']
        file_name = os.path.join(path, '..', 'models', 'log' + param_str + '.txt')
        # Catch print output of tensorflow functions
        f = io.StringIO()
        with redirect_stdout(f):
            print(str(model.evaluate(train_data, train_labels, verbose=2)))
            print('\n')
            print(str(model.evaluate(test_data, test_labels, verbose=2)))
            print('\n')
            print(str(model.summary()))
        out = f.getvalue()
        # Write output into logfile
        with open(file_name, 'w') as logfile:
            logfile.write(out)
            # logfile.write('\n')
    if plot:
        # Validate model
        test_predictions = model.predict(test_data, batch_size=parameters['batch_size']).flatten()
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.scatter(test_labels, test_predictions, alpha=0.05)
        ax.set_xlabel('True Values [MPG]')
        ax.set_ylabel('Predictions [MPG]')
        # lims = [min(test_labels), max(test_labels)]
        lims = [-0.2, 4/3+0.2]
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.plot(lims, lims)

        path = os.path.dirname(os.path.abspath(__file__))
        param_str = parameters['filename']
        file_name = os.path.join(path, '..', 'models', 'figures', 'fig' + param_str + '.png')
        fig.tight_layout()
        fig.savefig(file_name, dpi=150)
        # plt.show()
        plt.close()


def learning(filename, parameters, silent=False, regenerate=True, plot=True):
    # Read data
    [[train_labels, train_data], [test_labels, test_data]] = transform_data(filename)

    # scaler = MinMaxScaler()
    # train_labels = scaler.fit_transform(train_labels)
    # print(f'train_data: \n{train_data}')
    # print(f'train_labels: \n{train_labels}')
    parameters['filename'] = param_filename(parameters)

    # '''
    model = build_model(parameters, shape=test_data.shape[1])

    model = train_model(model, train_data, train_labels, parameters, silent, regenerate)

    validate_model(model, train_data, train_labels, test_data, test_labels, parameters, plot=plot)
    # validate_model(model, train_data, train_labels)
    # '''
