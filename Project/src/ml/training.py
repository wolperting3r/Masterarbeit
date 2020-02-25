import tensorflow as tf
from tensorflow import keras
from tqdm.keras import TqdmCallback

import os
import sys


def train_model(model, train_data, train_labels, parameters, silent):
    # Build tensorflow dataset
    dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))\
            .batch(parameters['batch_size'])\
            .prefetch(parameters['batch_size']*4)
    # Train Model
    # Early stopping callback
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='loss',
                                                            min_delta=10e-8,
                                                            patience=50,
                                                            verbose=0,
                                                            mode='auto',
                                                            baseline=None)
    # History to csv callback
    path = os.path.dirname(os.path.abspath(sys.argv[0]))
    param_str = parameters['filename']
    file_name = os.path.join(path, 'models', 'history', 'history' + param_str + '.csv')
    csv_logger = tf.keras.callbacks.CSVLogger(file_name, separator=',', append=False)

    # Train model
    model.fit(dataset,
              shuffle=True,
              epochs=parameters['epochs'],  # war 10000
              verbose=0,
              callbacks=[TqdmCallback(verbose=(0 if silent else 1)),
                         early_stopping_callback,
                         csv_logger])

    # Save model
    path = os.path.dirname(os.path.abspath(sys.argv[0]))
    param_str = parameters['filename']
    file_name = os.path.join(path, 'models', 'models', 'model' + param_str + '.h5')
    model.save(file_name)
    return model


def load_model(parameters):
    path = os.path.dirname(os.path.abspath(sys.argv[0]))
    param_str = parameters['filename']
    file_name = os.path.join(path, 'models', 'models', 'model' + param_str + '.h5')
    model = tf.keras.models.load_model(file_name)
    return model
