# from src.d.data_generation import generate_data
from src.d.data_generation import generate_data
from src.ml.machine_learning import learning, saving
from src.ml.utils import param_filename

# Suppress tensorflow logging
import logging
import os
from itertools import product as itpd
from multiprocessing import Process

# import threading
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def ml(
    plot,
    network,
    stencil,
    layer,
    activation,
    epochs=25,
    learning_rate=1e-3,
    neg=False,
    angle=False,
    rot=False,
    data=['circle'],
    smearing=False,
    hf='hf',
    hf_correction=False,
    dropout=0,
    plotdata=False,
    flip=False,
    cut=False,
    dshift=0,
    shift=0,
    bias=True,
    interpolate=0,
    edge=0,
    custom_loss=0,
    addstring=False,
):

    batch_size = 128
    equal_kappa = True

    # Parameters
    parameters = {
        'network': network,              # Network type
        'epochs': epochs,                # Number of epochs
        'layers': layer,                 # Autoencoder: [n*Encoder Layers, 1*Coding Layer, 1*Feedforward Layer]
        'stencil_size': stencil,         # Stencil size [x, y]
        'equal_kappa': equal_kappa,      # P(kappa) = const. or P(r) = const.
        'learning_rate': learning_rate,  # Learning Rate
        'batch_size': batch_size,        # Batch size
        'activation': activation,        # Activation function
        'negative': neg,                 # Negative values too or only positive
        'angle': angle,                  # Use the angles of the interface too
        'rotate': rot,                   # Rotate the data before learning
        'data': data,                    # 'ellipse', 'circle', 'both'
        'smear': smearing,               # Use smeared data
        'hf': hf,                        # Use height function
        'hf_correction': hf_correction,  # Use height function as input for NN
        'plotdata': plotdata,
        # 'dropout': dropout               # dropout fraction
        'flip': flip,
        'cut': cut,
        'dshift': dshift,
        'shift': shift,
        'bias': bias,
        'interpolate': interpolate,
        'edge': edge,
        'custom_loss': custom_loss,
        'addstring': addstring,
        #'addstring': '_dshift1b_shift_kappa',
    }

    # print(f'parameters:\n{parameters}')
    # Generate filename string
    parameters['filename'] = param_filename(parameters) + parameters['addstring']

    # Execute learning
    if parameters['network'] != 'auto':
        learning(parameters, silent=True, plot=plot)

    elif parameters['network'] == 'auto':
        if not plot:
            parameters['network'] = 'autoencdec'
            parameters['epochs'] = int(parameters['epochs']*2)
            learning(parameters, silent=True, plot=plot)
            parameters['epochs'] = int(parameters['epochs']/2)
        parameters['network'] = 'autoenc'
        learning(parameters, silent=True, plot=plot)

    parameters = None


def save(
    plot,
    network,
    stencil,
    layer,
    activation,
    epochs=25,
    learning_rate=1e-3,
    neg=False,
    angle=False,
    rot=False,
    data=['circle'],
    smearing=False,
    hf='hf',
    hf_correction=False,
    dropout=0,
    plotdata=False,
    flip=False,
    cut=False,
    dshift=0,
    shift=0,
    bias=True,
    interpolate=0,
    edge=0,
    custom_loss=0,
    addstring=False,
):

    batch_size = 128
    equal_kappa = True

    # Parameters
    parameters = {
        'network': network,              # Network type
        'epochs': epochs,                # Number of epochs
        'layers': layer,                 # Autoencoder: [n*Encoder Layers, 1*Coding Layer, 1*Feedforward Layer]
        'stencil_size': stencil,         # Stencil size [x, y]
        'equal_kappa': equal_kappa,      # P(kappa) = const. or P(r) = const.
        'learning_rate': learning_rate,  # Learning Rate
        'batch_size': batch_size,        # Batch size
        'activation': activation,        # Activation function
        'negative': neg,                 # Negative values too or only positive
        'angle': angle,                  # Use the angles of the interface too
        'rotate': rot,                   # Rotate the data before learning
        'data': data,                    # 'ellipse', 'circle', 'both'
        'smear': smearing,               # Use smeared data
        'hf': hf,                        # Use height function
        'hf_correction': hf_correction,  # Use height function as input for NN
        # 'dropout': dropout               # dropout fraction
        'plotdata': plotdata,
        'flip': flip,
        'cut': cut,
        'dshift': dshift,
        'shift': shift,
        'bias': bias,
        'interpolate': interpolate,
        'edge': edge,
        'custom_loss': custom_loss,
        'addstring': addstring,
    }

    # Generate filename string
    parameters['filename'] = param_filename(parameters)

    # Execute learning
    saving(parameters)


    parameters = None



def exe_dg(**kwargs):
    print(f'kwargs:\n{kwargs}')
    # Sort input keyword arguments
    order = ['N_values', 'stencils', 'ek', 'neg', 'silent', 'geometry', 'smearing', 'usenormal', 'interpolate']
    kwargs = {k: kwargs[k] for k in order}
    # Create job list according to input arguments
    job_list = list(itpd(*kwargs.values()))
    if len(job_list) > 1:
        # Execute job list with multithreading
        jobs = []
        [jobs.append(Process(target=generate_data, args=job)) for job in job_list]
        [j.start() for j in jobs]
        [j.join() for j in jobs]
    else:
        # Execute job
        generate_data(**dict(zip(kwargs.keys(), job_list[0])))


def exe_ml(**kwargs):
    # Sort input keyword arguments
    order = ['plot', 'network', 'stencil', 'layer', 'activation', 'epochs', 'learning_rate', 'neg', 'angle', 'rot', 'data', 'smearing', 'hf', 'hf_correction', 'dropout', 'plotdata', 'flip', 'cut', 'dshift', 'shift', 'bias', 'interpolate', 'edge', 'custom_loss', 'addstring',]
    kwargs = {k: kwargs[k] for k in order}
    # Execute machine learning
    plot = kwargs.get('plot')
    if not plot[0]: # Execute training job list with multithreading
        kwargs['plotdata'] = [False]
        jobs = []
        for job in list(itpd(*kwargs.values())):
            jobs.append(Process(target=ml, args=job))
        [j.start() for j in jobs]
        [j.join() for j in jobs]
    elif plot[0]:
        # Execute validation job list with multithreading
        for job in list(itpd(*kwargs.values())):
            ml(**dict(zip(kwargs.keys(), job)))


def exe_save(**kwargs):
    # Sort input keyword arguments
    order = ['plot', 'network', 'stencil', 'layer', 'activation', 'epochs', 'learning_rate', 'neg', 'angle', 'rot', 'data', 'smearing', 'hf', 'hf_correction', 'dropout', 'plotdata', 'flip', 'cut', 'dshift', 'shift', 'bias', 'interpolate', 'edge', 'custom_loss', 'addstring',]
    kwargs = {k: kwargs[k] for k in order}
    # Execute saving job list with multithreading
    for job in list(itpd(*kwargs.values())):
        save(**dict(zip(kwargs.keys(), job)))
