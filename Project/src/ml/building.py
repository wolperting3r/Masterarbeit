import tensorflow as tf
from tensorflow.keras import layers

import os
import sys


def create_model(parameters, shape):
    '''Create different tensorflow models'''
    # Classical feedforward network
    if parameters['network'] == 'mlp':
        model = tf.keras.Sequential()
        # Add feedforward layers defined in parameters['layers']
        model.add(layers.InputLayer(input_shape=(shape[1],)))
        for l in parameters['layers']:
            model.add(layers.Dense(l, activation=parameters['activation'], use_bias=parameters['bias']))
            # if parameters['dropout'] > 0:
                # model.add(layers.Dropout(parameters['dropout']))
        model.add(layers.Dense(1, activation='linear'))

    # Convolutional network
    elif parameters['network'] == 'cvn':
        model = tf.keras.Sequential()
        if not parameters['hf_correction']:
            # Add one input layer
            model.add(layers.InputLayer(input_shape=(shape[1], shape[2], 1)))
        else:
            # Add two input layers (one additional for curvature value calculated with height function)
            model.add(layers.InputLayer(input_shape=(shape[1], shape[2], 2)))
        # Add convolutional layers as defined in parameters['layers']
        for l in parameters['layers']:
            model.add(layers.Conv2D(l, (2, 2), activation=parameters['activation']))
            # if parameters['dropout'] > 0:
                # model.add(layers.Dropout(parameters['dropout']))
        # Add flatten layer and final layers
        model.add(layers.Flatten())
        model.add(layers.Dense(100, activation=parameters['activation']))
        model.add(layers.Dense(1, activation='linear'))

    # Full autoencoder (to train encoder part up to coding layer)
    elif parameters['network'] == 'autoencdec':
        model = tf.keras.Sequential()
        model.add(layers.InputLayer(input_shape=(shape[1],)))
        # Encoder
        for l in parameters['layers'][:-2]:
            model.add(layers.Dense(l, activation=parameters['activation']))
        # Coding layer
        model.add(layers.Dense(parameters['layers'][-2], activation='linear'))
        # Decoder
        for l in reversed(parameters['layers'][:-2]):
            model.add(layers.Dense(l, activation=parameters['activation']))
        # Output layer
        model.add(layers.Dense(shape[1], activation='linear'))

    # Autoencoder (reuse encoder from full autoencoder)
    elif parameters['network'] == 'autoenc':
        # Import encoder model
        parent_path = os.path.dirname(os.path.abspath(sys.argv[0]))
        # Create file name string with autoencdec
        param_str = parameters['filename']
        file_name = os.path.join(parent_path, 'models', 'models', 'model' + param_str + '.h5')
        # Load model
        auto_model = tf.keras.models.load_model(file_name)
        # Pop decoder layers
        auto_model.pop()
        for i in parameters['layers'][:-2]:
            auto_model.pop()
        # Make decoder untrainable
        auto_model.trainable = False
        # Build new model
        model = tf.keras.Sequential([
            auto_model,
            tf.keras.layers.Dense(parameters['layers'][-1], activation=parameters['activation']),
            tf.keras.layers.Dense(1, activation='linear')
        ])

    return model

def custom_loss(y_true, y_pred):
    return tf.math.reduce_mean(tf.math.multiply(tf.math.squared_difference(y_true, y_pred),tf.math.square(tf.math.multiply(tf.math.subtract(0.44, tf.math.abs(y_true)), 2)) ))
    #return tf.math.subtract(tf.abs(y_true),0.5)


def build_model(parameters, shape):
    # Create tensorflow model
    model = create_model(parameters, shape)
    # Compile model with optimizer and loss function
    model.compile(optimizer=tf.keras.optimizers.Adam(parameters['learning_rate']),
                  loss='mse',
                  metrics=['mae', 'mse'])
    # model.compile(optimizer=tf.keras.optimizers.Adam(parameters['learning_rate']),
                  # loss= custom_loss,   # custom_loss oder 'mse'
                  # metrics=['mae', 'mse'])
    # Print summary
    for key, value in parameters.items():
        print(f'{key}:\t\t{value}')
    model.summary()
    return model
