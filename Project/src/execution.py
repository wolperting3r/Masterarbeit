# from src.d.data_generation import generate_data
from src.d.data_generation import generate_data
from src.ml.machine_learning import learning
from src.ml.utils import param_filename

# Suppress tensorflow logging
import logging
import os
import itertools
import multiprocessing
import threading

# import threading
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def gendat(st_sz, equal_kappa=True, neg=False, N_values=1, silent=True, ellipse=False):  # 1e6
    print(f'Generating data:\nEllipse:\t{ellipse}\nStencil:\t{st_sz}\nKappa:\t\t{equal_kappa}\nNeg. Values\t{neg}\nN_values:\t{int(N_values)}')
    generate_data(N_values=N_values, st_sz=st_sz, equal_kappa=equal_kappa, neg=neg, silent=False, ellipse=ellipse)
    # generate_data(N_values=1e6, st_sz=inzip[0], equal_kappa=inzip[1], neg=inzip[2], silent=True)
    # print(f'Generation finished: {inzip}')


def ml(network, stencil, layer, act, plot, epochs=25, batch_size=128, learning_rate=1e-3, equal_kappa=True, neg=False, angle=False, rot=False, data=['circle']):
    # Parameters
    parameters = {'network': network,       # Network type
                  'epochs': epochs,         # Number of epochs
                  'layers': layer,  # Autoencoder: [n*Encoder Layers, 1*Coding Layer, 1*Feedforward Layer]
                  'stencil_size': stencil,  # Stencil size [x, y]
                  'equal_kappa': equal_kappa, # P(kappa) = const. or P(r) = const.
                  'learning_rate': learning_rate,
                  'batch_size': batch_size, # Batch size
                  'activation': act,        # Activation function
                  'negative': neg,          # Negative values too or only positive
                  'angle': angle,           # Use the angles of the interface too
                  'rotate': rot,          # Rotate the data before learning
                  'data': data,             # 'ellipse', 'circle', 'both'
                 }
    # Generate filename string
    parameters['filename'] = param_filename(parameters)

    '''
    # Print start string
    print('\nParameters:')
    for key, value in parameters.items():
        print(str(key) + ': ' + str(value))
    '''

    # Execute learning
    if parameters['network'] != 'auto':
        learning(parameters, silent=True, plot=plot)
    elif parameters['network'] == 'auto':
        if plot == False:
            parameters['network'] = 'autoencdec'
            parameters['epochs'] = int(parameters['epochs']*2)
            learning(parameters, silent=True, plot=plot)
            parameters['epochs'] = int(parameters['epochs']/2)
        parameters['network'] = 'autoenc'
        learning(parameters, silent=True, plot=plot)

    ''' 
    # Print finished string
    print('\nFinished:')
    for key, value in parameters.items():
        print(str(key) + ': ' + str(value))
    print('\n')
    '''

    parameters = None

def exe_dg(stencils, ek, neg, N_values, silent, ellipse):
    # Execute data generation
    job_list = list(itertools.product(*(stencils, ek, neg, N_values, silent, ellipse)))
    if len(job_list)>1:
        jobs = []
        for job in job_list:
            process = multiprocessing.Process(target=gendat, args=job)
            jobs.append(process)
        for j in jobs:
            j.start()
        for j in jobs:
            j.join()
    else:
        gendat(job_list[0][0], job_list[0][1], job_list[0][2], job_list[0][3], job_list[0][4], job_list[0][5])


def exe_ml(network, stencils, layers, activation=['relu'], epochs=[25], batch_size=[128], learning_rate=[1e-3], equal_kappa=[True], neg=False, angle=False, rot=False, data=['circle']):
    # Execute machine learning
    plot = [False]
    job_list = list(itertools.product(*(network, stencils, layers, activation, plot, epochs, batch_size, learning_rate, equal_kappa, neg, angle, rot, data)))
    print(f'job_list:\n{job_list}')

    jobs = []
    for job in job_list:
        process = multiprocessing.Process(target=ml, args=job)
        jobs.append(process)
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()

def exe_ml_plot(network, stencils, layers, activation=['relu'], epochs=[25], batch_size=[128], learning_rate=[1e-3], equal_kappa=[True], neg=False, angle=False, rot=False, data=['circle']):
    # Plot
    plot = [True]
    job_list = list(itertools.product(*(network, stencils, layers, activation, plot, epochs, batch_size, learning_rate, equal_kappa, neg, angle, rot, data)))
    for job in job_list:
        ml(job[0], job[1], job[2], job[3], job[4], job[5], job[6], job[7], job[8], job[9], job[10], job[11], job[12])
