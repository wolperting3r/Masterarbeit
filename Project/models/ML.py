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

from data.data_transformation import transform_data

# Enable full width output for numpy (https://stackoverflow.com/questions/43514106/python-terminal-output-width)
np.set_printoptions(suppress=True, linewidth=250, threshold=250)

def relative_mse(y_true, y_pred):
    return tf.reduce_mean(tf.math.squared_difference(y_pred,y_true))/y_true


def build_model(shape):
    # Build keras model
    model = tf.keras.Sequential([
        layers.Dense(100, activation='relu', input_shape=(shape,)),
        layers.Dense(80, activation='relu'),
        layers.Dense(80, activation='relu'),
        layers.Dense(1, activation='linear')
    ])

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='mse',
                  metrics=['mae', 'mse', relative_mse])
    return model


def train_model(model, train_data, train_labels, regenerate=True):
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
        model.fit(dataset,
                  shuffle=True,
                  epochs=5000,  # war 10000
                  verbose=0,
                  callbacks=[TqdmCallback(verbose=1),
                            early_stopping_callback])

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


def learning():
    # Read data
    [[train_labels, train_data], [test_labels, test_data]] = transform_data()

    # scaler = MinMaxScaler()
    # train_labels = scaler.fit_transform(train_labels)
    # print(f'train_data: \n{train_data}')
    # print(f'train_labels: \n{train_labels}')
    # '''
    model = build_model(shape=test_data.shape[1])

    model = train_model(model, train_data, train_labels)

    validate_model(model, test_data, test_labels)
    # validate_model(model, train_data, train_labels)
    # '''
