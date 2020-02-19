from src.data_generation import generate_data
from src.machine_learning import learning

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


def ml(network, stencil, layer, act, plot):
    # Parameters
    parameters = {'network': network,
                  'epochs': 25,
                  'layers': layer,
                  'stencil_size': stencil,
                  'equal_kappa': True,
                  'learning_rate': 1e-3,
                  'batch_size': 128,
                  'activation': act}

    # Get data filename
    print('\nParameters:')
    for key, value in parameters.items():
        print(str(key) + ': ' + str(value))
    filename = 'data_' + \
            str(parameters['stencil_size'][0]) + 'x' + str(parameters['stencil_size'][1]) + '_' + \
            ('eqk' if parameters['equal_kappa'] else 'eqr') + \
            '.feather'

    print('\nImporting ' + filename)
    # Execute learning
    learning(filename, parameters, silent=True, regenerate=(False if plot else True), plot=plot)
    print('Finished:')
    for key, value in parameters.items():
        print(str(key) + ': ' + str(value))



''' Data Generation '''
'''
stencils = [[3, 3], [5, 5], [7, 3], [3, 7]]
ek = [True, False]
job_list = list(itertools.product(*(stencils, ek)))

jobs = []
for job in job_list:
    process = multiprocessing.Process(target=gendat, args=[job])
    jobs.append(process)
for j in jobs:
    j.start()
for j in jobs:
    j.join()
# '''


''' Machine Learning '''
# MLP
# '''
plot = [False]

# relu
stencils = [[3, 3], [5, 5], [7, 3], [3, 7]]
layers = [[100], [100, 80], [100, 80, 50]]
act = ['relu']
network = ['mlp']
job_list = list(itertools.product(*(stencils, layers, act, plot)))
jobs = []
for job in job_list:
    process = multiprocessing.Process(target=ml, args=job)
    jobs.append(process)
for j in jobs:
    j.start()
for j in jobs:
    j.join()

# tanh
stencils = [[3, 3], [5, 5], [7, 3], [3, 7]]
layers = [[100]]
act = ['tanh']
job_list = list(itertools.product(*(stencils, layers, act, plot)))
jobs = []
for job in job_list:
    process = multiprocessing.Process(target=ml, args=job)
    jobs.append(process)
for j in jobs:
    j.start()
for j in jobs:
    j.join()


plot = [True]

# relu
stencils = [[3, 3], [5, 5], [7, 3], [3, 7]]
layers = [[100], [100, 80], [100, 80, 50]]
act = ['relu']
job_list = list(itertools.product(*(stencils, layers, act, plot)))
for job in job_list:
    ml(job[0], job[1], job[2], job[3])

# tanh
stencils = [[3, 3], [5, 5], [7, 3], [3, 7]]
layers = [[100]]
act = ['tanh']
job_list = list(itertools.product(*(stencils, layers, act, plot)))
for job in job_list:
    ml(job[0], job[1], job[2], job[3])
# '''

# CVN
stencils = [[3, 3], [5, 5]]
layers = [[64, 128]]
activation = ['relu']
plot = [False]
job_list = list(itertools.product(*(stencils, layers, activation, plot)))

print(f'job_list:\n{job_list}')

jobs = []
for job in job_list:
    process = multiprocessing.Process(target=ml, args=job)
    jobs.append(process)
for j in jobs:
    j.start()
for j in jobs:
    j.join()

plot = [True]
job_list = list(itertools.product(*(stencils, layers, activation, plot)))
for job in job_list:
    ml(job[0], job[1], job[2], job[3])
