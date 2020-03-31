# MACHINE LEARNING
import numpy as np

from src.d.data_transformation import transform_data
from src.ml.building import build_model
from src.ml.training import train_model, load_model
from src.ml.validation import validate_model_loss, validate_model_plot

# Enable full width output for numpy (https://stackoverflow.com/questions/43514106/python-terminal-output-width)
np.set_printoptions(suppress=True, linewidth=250, threshold=250)


def learning(parameters, silent=False, plot=True):
    # Get data (reshape if network is convolutional network)
    [[train_labels, train_data, train_angle, train_kappa],
     [test_labels, test_data, test_angle, test_kappa],
     [val_labels, val_data, val_angle, val_kappa]] = transform_data(
         parameters,
         reshape=(True if parameters['network'] == 'cvn' else False),
         plot=plot
     )  # kappa = 0 if parameters['hf'] == False
    '''
    ind = 0
    print_data_grad = test_data.transpose((0, 1, 3, 2))[ind]
    # print_data_grad = test_data[ind]
    print(f'\nGedreht:\n{print_data_grad}')
    # '''
    # '''
    # Make output = input to train autoencoder
    if parameters['network'] == 'autoencdec':
        train_labels = train_data
        test_labels = test_data

    if not plot:
        # Build model
        model = build_model(parameters, train_data.shape)
        # Train model
        model = train_model(model, train_data, train_labels, val_data, val_labels, parameters, silent)
        # Validate model
        validate_model_loss(model, train_data, train_labels, test_data, test_labels, parameters)
    else:
        # Load model
        model = load_model(parameters)
        # Create validation plot
        validate_model_plot(model, test_data, test_labels, parameters, test_kappa=test_kappa[:, 0])
    # '''
