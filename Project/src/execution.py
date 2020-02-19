from src.d.data_generation import generate_data
from src.ml.machine_learning import learning
from src.ml.utils import param_filename

# Suppress tensorflow logging
import logging
import os
import itertools
import multiprocessing

# import threading
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def gendat(inzip):
    print(f'Generating data: {inzip}')
    generate_data(N_values=1e6, st_sz=inzip[0], equal_kappa=inzip[1], silent=True)
    print(f'Generation finished: {inzip}')


def ml(network, stencil, layer, act, plot, epochs=25, batch_size=128, learning_rate=1e-3, equal_kappa=True):
    # Parameters
    parameters = {'network': network,
                  'epochs': epochs,
                  'layers': layer,  # Autoencoder: [n*Encoder Layers, 1*Coding Layer, 1*Feedforward Layer]
                  'stencil_size': stencil,
                  'equal_kappa': equal_kappa,
                  'learning_rate': learning_rate,
                  'batch_size': batch_size,
                  'activation': act
                 }
    # Generate filename string
    parameters['filename'] = param_filename(parameters)

    # Print start string
    print('\nParameters:')
    for key, value in parameters.items():
        print(str(key) + ': ' + str(value))

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

    # Print finished string
    print('Finished:')
    for key, value in parameters.items():
        print(str(key) + ': ' + str(value))

    parameters = None

def exe_dg(stencils, ek):
    # Execute data generation
    job_list = list(itertools.product(*(stencils, ek)))
    jobs = []
    for job in job_list:
        process = multiprocessing.Process(target=gendat, args=[job])
        jobs.append(process)
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()

def exe_ml(network, stencils, layers, activation=['relu'], epochs=[25], batch_size=[128], learning_rate=[1e-3], equal_kappa=[True]):
    # Execute machine learning
    plot = [False]
    job_list = list(itertools.product(*(network, stencils, layers, activation, plot, epochs, batch_size, learning_rate, equal_kappa)))
    print(f'job_list:\n{job_list}')

    jobs = []
    for job in job_list:
        process = multiprocessing.Process(target=ml, args=job)
        jobs.append(process)
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()

def exe_ml_plot(network, stencils, layers, activation=['relu'], epochs=[25], batch_size=[128], learning_rate=[1e-3], equal_kappa=[True]):
    # Plot
    plot = [True]
    job_list = list(itertools.product(*(network, stencils, layers, activation, plot, epochs, batch_size, learning_rate, equal_kappa)))
    for job in job_list:
        map(ml, [job])
        # ml(job[0], job[1], job[2], job[3], job[4], job[5], job[6], job[7], job[8])
