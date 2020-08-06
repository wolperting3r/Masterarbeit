import os
import sys
import re
import pandas as pd
import numpy as np
from progressbar import *
from scipy import ndimage
import tikzplotlib as tkz


import matplotlib.pyplot as plt
import matplotlib.colors as colors

import itertools
from multiprocessing import Process

# Enable full width output for numpy (https://stackoverflow.com/questions/43514106/python-terminal-output-width)
np.set_printoptions(suppress=True, linewidth=np.inf, threshold=np.inf)


def tecplot2data(f, oszb, st_sz, filtering, filter):
    print(f'Generating data from {f} with stencil {st_sz[0]}x{st_sz[1]}')
    if os.path.isfile(os.path.join(f, 'res', 'oscillation.dat')):
        file_name = os.path.join(f, 'res', 'oscillation.dat')
    elif os.path.isfile(os.path.join(f, 'res', 'staticbubble.dat')):
        file_name = os.path.join(f, 'res', 'staticbubble.dat')

    with open(file_name, 'r') as myfile:
        data = myfile.read()
        # Append 'zone t' to file for capturing blocks later
        data = data + '\nZONE T'
        # Get variables
        variables = re.split(r'"\n"', re.search(r'(?<=VARIABLES = ")\w+("\n"\w+)+(?=")', data)[0])
        # Bei StaticBubble müssen die nächsten beiden Zeilen auskommentiert werden
        # variables.remove('X')
        # variables.remove('Y')
        n_vars = len(variables)
        # Get gs
        gs = [int(i) for i in re.findall(r'\d+', re.search(r'I=\d+, J=\d+, K=\d+', data)[0])]
        length = gs[0]**2
        # Get all timesteps (blocks)
        blocks = re.findall(r'ZONE\sT[\d\D]+?(?=ZONE\sT)', data)
        print(f'len(blocks):\t{len(blocks)}')
        # Remove first block (no information)
        # blocks = blocks[1:]

        # Get x/y from first block 
        # coordinates = {}
        block = blocks[0]
        numbers = np.array(re.findall(r'(\-?\d\.\d+E[\+\-]\d{2})', block))
        # print(f'len(numbers):\t{len(numbers)}')
        coordinates = np.empty((2, gs[0], gs[1], gs[2]))
        # Get x coordinates
        coordinates[0, :, :, :] = np.reshape(numbers[:np.prod(gs)], (gs[0], gs[1], gs[2]))
        # Get y coordinates
        coordinates[1, :, :, :] = np.reshape(numbers[np.prod(gs):2*np.prod(gs)], (gs[0], gs[1], gs[2]))
        max_x = np.max(coordinates[0, :, :, :])
        max_y = np.max(coordinates[1, :, :, :])
        delta = max_x/(gs[0]-1)
        coordinates = np.reshape(coordinates[:, :, :, 0], (2, gs[0], gs[1]))

        # Coordinates are messed up, make own coordinate matrix
        coordinates = np.empty((2, gs[0]-1, gs[1]-1))
        cord_vec = np.reshape(np.array(range(0, gs[0]-1)*delta), (gs[0]-1, 1))
        coordinates[0, :, :] = cord_vec.T
        coordinates[1, :, :] = np.flip(cord_vec)

        # print('Blocks abgeschnitten!')
        # blocks = blocks[:1]
        print(f'len(numbers):\t{len(numbers)}')
        print(f'2*np.prod(gs):\t{2*np.prod(gs)}')

        values = np.empty((n_vars-2, gs[0]-1, gs[1]-1))
        for v in variables:
            # j = 0: concentration, j = 1: curvature
            j = variables.index(v)
            # Assign next 128*128 values to variable v
            if j >= 2:
                values[j-2, :, :] = np.reshape(numbers[
                    2*np.prod(gs)+(j-2)*(gs[0]-1)*(gs[1]-1) : 2*np.prod(gs)+(j-2)*(gs[0]-1)*(gs[1]-1)+(gs[0]-1)*(gs[1]-1)
                ], (gs[0]-1, gs[1]-1))
        if (0 == 1): # no export
            fig, ax = plt.subplots()
            ax.imshow(values[0, :, :], cmap='Greys_r')
            # ax.imshow(values[1, :, :], cmap='viridis', norm=plt.Normalize(-30, 100))
            plt.show()
        else:
            # Make figure without border
            fig = plt.figure(frameon=False)
            fig.set_size_inches(10,10)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ''' # Artefakt
            values = np.rot90(values, k=3, axes=(1, 2)) 
            sqrsize = 45
            xlower = 17
            ylower = 68
            # '''
            ''' # Zebra
            values = np.rot90(values, k=1, axes=(1, 2)) 
            sqrsize = 45
            xlower = 10
            ylower = 42
            # '''
            # ''' # Falsche Werte
            # values = np.rot90(values, k=1, axes=(1, 2)) 
            sqrsize = 128
            xlower = 0
            ylower = 0
            # '''
            limits = [[xlower, xlower+sqrsize], [ylower, ylower+sqrsize]] # x, y
            y, x = np.meshgrid(np.linspace(limits[1][0], limits[1][1], limits[1][1]-limits[1][0]), np.linspace(limits[0][0], limits[0][1], limits[0][1]-limits[0][0]))
            print(f'x.shape:\t{x.shape}')
            # Krümmung oszillierende Blase (-30 - 100)
            # pcm = ax.pcolormesh(y, x, values[1, limits[0][0]:limits[0][1], limits[1][0]:limits[1][1]], cmap='RdBu', norm=colors.TwoSlopeNorm(vmin=-30, vcenter=0, vmax=100))
            # Krümmung statische Blase (-0.3 - 1)
            pcm = ax.pcolormesh(y, x, values[1, limits[0][0]:limits[0][1], limits[1][0]:limits[1][1]], cmap='RdBu', norm=colors.TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=1))

            # Konzentration verzerrte Skala
            # pcm = ax.pcolormesh(y, x, values[0, limits[0][0]:limits[0][1], limits[1][0]:limits[1][1]], cmap='Greys_r', norm=colors.TwoSlopeNorm(vmin=0, vcenter=0.1, vmax=1))
            # Konzentration lineare Skala
            # pcm = ax.pcolormesh(y, x, values[0, limits[0][0]:limits[0][1], limits[1][0]:limits[1][1]], cmap='Greys_r', norm=colors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1))

            # tkz.save('result2d.tex', axis_height='7cm', axis_width='7cm') 
            # '''
            plt.savefig('result2d.eps')

            with open('result2d.eps', 'r') as myfile:
                data = myfile.read()
                data = re.sub(r'fill', 'gsave fill grestore stroke', data)


            with open('result2d.eps', 'w') as myfile:
                myfile.write(data)
            # '''
            plt.show()



gridsize = 128
files = [
    # '2007201919 ml falsche Werte',
    # '2008031037 Edge neu Relaxation Gauss Ellipse s1',
    # '2008031037 Edge neu Relaxation Gauss Ellipse s1',
    # '2005181419 c<01 c>99 abgeschnitten artefakte',
    # '2006041146 dshift1 0.05 0.95 gewichtung mit 141 stencil streifenmuster',
    # '2008041519 staticBubble 128 ml sharp',
    # '2008031931 Edge neu Relaxation FNB Ellipse s15 15s',
    # f'2007181652 staticBubble {gridsize} cvofls',
    f'2007181652 staticBubble {gridsize} cds mit w+g',
]

# st_sz = [[5, 5], [7, 7], [9, 9]]
st_sz = [[7, 7]]
# st_sz = [[5, 5]]

filtering = [0]
filter = ['g']

oszb = [False]

kwargs = {'f': files, 'oszb': oszb, 'st_sz': st_sz, 'filtering': filtering, 'filter': filter}

job_list = list(itertools.product(*kwargs.values()))
# '''
# Single Process (for plotting)
for job in job_list:
    tecplot2data(**dict(zip(kwargs.keys(), job)))
# '''
'''
# Multiprocessing
jobs = []
[jobs.append(Process(target=tecplot2data, args=job)) for job in job_list]
[j.start() for j in jobs]
[j.join() for j in jobs]
# '''
