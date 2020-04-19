# MACHINE LEARNING
import numpy as np

from src.d.data_transformation import transform_data, transform_kappa
from src.ml.building import build_model
from src.ml.training import train_model, load_model
from src.ml.validation import validate_model_loss, validate_model_plot
from src.ml.utils import param_filename
import os
import sys

# Enable full width output for numpy (https://stackoverflow.com/questions/43514106/python-terminal-output-width)
np.set_printoptions(suppress=True, linewidth=250, threshold=250)


def learning(parameters, silent=False, plot=True): # Get data (reshape if network is convolutional network)
    if plot:
        param_tmp = parameters.copy()
        # Set data to plotdata to load correct data file for plotting
        param_tmp['data'] = param_tmp['plotdata']
        param_tmp['filename'] = param_filename(param_tmp, plotdata_as_data=True)
    else:
        param_tmp = parameters
    # '''
    [[train_labels, train_data, train_angle],
     [test_labels, test_data, test_angle],
     [val_labels, val_data, val_angle]] = transform_data(
         param_tmp,
         reshape=(True if parameters['network'] == 'cvn' else False),
         plot=plot
     )  # kappa = 0 if parameters['hf'] == False
    # '''

    # '''
    if plot:
        [[train_k_labels, train_kappa], [test_k_labels, test_kappa], [val_k_labels, val_kappa]] = transform_kappa(
             param_tmp,
             reshape=(True if parameters['network'] == 'cvn' else False),
             plot=plot
         )  # kappa = 0 if parameters['hf'] == False
    else:
        print(f'Training data: {train_data.shape}')
    # '''
    '''
    ind = 0
    print_data_grad = test_data.transpose((0, 1, 3, 2))[ind]
    # print_data_grad = test_data[ind]
    print(f'\nGedreht:\n{print_data_grad}')
    # '''
    '''
    # Make output = input to train autoencoder
    if parameters['network'] == 'autoencdec':
        train_labels = train_data
        test_labels = test_data
    # '''

    # '''
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

        save_to_txt(model, parameters)

        # Create validation plot
        # validate_model_plot(model, test_data, test_labels, parameters, test_kappa=test_kappa[:, 0], test_k_labels=test_k_labels)
    # '''

def saving(parameters):
    # Load model
    model = load_model(parameters)

    # Get export path
    param_str = parameters['filename']
    path = os.path.dirname(os.path.abspath(sys.argv[0]))
    file_name = os.path.join(path, 'model' + param_str + '.txt')

    # Reshape matrices to 1D vectors
    output_weights = []
    for w in model.get_weights():
        output_weights.append(np.reshape(w, ((w.shape[0]*w.shape[1] if len(w.shape) == 2 else w.shape[0]), 1)))

    # Export network to text file
    with open(file_name, 'w') as output:
        # Write number of input nodes
        output.write(str(parameters['stencil_size'][0]*parameters['stencil_size'][1])+'\n')
        # Write number of layers
        output.write(str(len(parameters['layers'])+1)+'\n')
        # Write nodes per layer
        for l in parameters['layers']:
            output.write(str(l)+'\n')
        output.write('1\n')
        # Write activation function per layer
        for l in range(len(parameters['layers'])):
            output.write(parameters['activation']+'\n')
        output.write('line\n')
        # Write 1 for bias_on and out_bias_on
        # output.write('1\n1\n')
        for w in output_weights:
            output.write("\n".join(map(str, w[:, 0])))
            output.write('\n')
