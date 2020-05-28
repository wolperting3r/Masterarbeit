# DATA GENERATION
import numpy as np
import itertools
import time
import pandas as pd
from progressbar import *
import matplotlib.pyplot as plt
import os

from .utils import (
    gt,
    plot_circle,
    plot_vof
)

'''
Outline:
    1. U = unif[0, 1]
    2. R = R_min + U(R_min - R_max) (R_min/Delta = 4, R_max/Delta = 1000)
    3. _x_c = _X_c + _U*Delta (random Center)
    4. theta = 2*pi*U, psi = arccos(2*U-1) (evenly distributed random angles across sphere)
    5. _x = (x_c+R*cos(theta)sin(psi), y_c+R*sin(theta)sin(psi), z_c+R*cos(psi))^T
    6. i, j, k (cell coordinates)
    7. F_(l, m, n) = VoF in 3x3x3 stencil, kappa = 2/R
    8. sign = U-0.5 -> F = (1-F if sign<0, F if sign>=0), kappa = (-kappa if sign<0, kappa if sign>=0)
    9. Store 27 VoF values and kappa
'''


def u(low=0.0, high=1.0):
    return np.random.uniform(low=low, high=high)


def axismatrix(n_grid, axis):
    # Generate matrix where values increase along the given axis starting at 0
    n_grid = int(n_grid)
    # Ascending values in z-direction
    return_matrix = np.tile(np.array(range(n_grid))/n_grid, (n_grid, 1))
    if axis == 0:
        return return_matrix.transpose((1, 0))
    elif axis == 1:
        return return_matrix


def generate_data(N_values, st_sz: [int, int]):

    time0 = time.time()

    # ↓↓ Parameters ↓↓ #
    # Script
    # N_values = 100000       # Number of values
    N_values = int(N_values)
    visualize = True if (N_values == 1) else False
    # Grid
    Delta = 1/1000          # Gridsize
    Delta_vof = 1/32        # VoF Gridsize 0.1% accuracy: 32 -> ~1000 (1024) points
    L = 1                   # Length of space
    X_c = np.array([L/2, L/2])   # Midpoint of geometry
    # Geometry
    R_min = max(st_sz)/2*Delta      # Minimal radius (so the circle is not smaller than the stencil)
    R_max = 0.5                     # Maximal radius
    kappa_min = L*Delta*2/R_min
    kappa_max = L*Delta*2/R_max
    # Stencil
    st_sz = [5, 5]      # Stencil size y, x
    # ↑↑ Parameters ↑↑ #
    
    # Calculate midpoints
    st_mid = [int((st_sz[0]-1)/2), int((st_sz[1]-1)/2)]
    # Generate stencil points relative to cell
    st_stp = np.array([np.arange(-st_mid[0], (st_mid[0]+1)),
                       np.arange(-st_mid[1], (st_mid[1]+1))])*Delta
    # Generate a local grid for later usage (y from top to bottom, x from left to right)
    local_grid = np.array([axismatrix(1/Delta_vof, 0),
                           axismatrix(1/Delta_vof, 1)])*Delta
    # Initialize output list
    output_list = []
    # Initialize progress bar
    widgets = ['Data generation: ', Percentage(), ' ', Bar(marker='=',left='[',right=']'),
           ' ', ETA()] #see docs for other options
    pbar = ProgressBar(widgets=widgets, maxval=N_values)
    pbar.start()
    for n in range(N_values):
        pbar.update(n)
        # Get random curvature
        curvature = kappa_min + u()*(kappa_max - kappa_min)
        # Calculate radius
        r = L*Delta*2/curvature
        '''
        # Get random radius
        r = R_min + u()*(R_max - R_min)
        # Calculate curvature
        curvature = L*Delta*2/r  # Stimmt das auch, wenn der Stencil anders gewählt wird?
        '''
        # Move midpoint by random amount inside one cell
        x_c = np.array([u(), u()])*Delta
        # Get random spherical angle
        theta = 2*np.pi*u()
        # Get cartesian coordinates on sphere surface
        x_rel = np.array([r*np.sin(theta),   # y
                          r*np.cos(theta)])  # x
        x = np.array([x_c[0]+x_rel[0],
                      x_c[1]+x_rel[1]])
        # Round point to get origin of local coordinates in global coordinates relative to geometry origin
        round_point = np.floor(x*1/Delta*L)*Delta/L
        '''
        1.: VOF Werte auf dem Kreuz finden
        2.: Gradienten auf dem Kreuz auswerten
        3.: Orientierung des Stencils bestimmen
        4.: Restliche VOF-Werte des Stencils bestimmen
        '''
        # Get origins of local coordinates of stencil points
        local_origins = np.array([round_point[0]+st_stp[0],
                                  round_point[1]+st_stp[1]])
        # Get list of all origins in stencil
        local_origins_list = np.array(list(itertools.product(*local_origins)))
        # Get list of all stencil indices combinations
        iter_loc_list = np.array(list(itertools.product(range(st_sz[0]),range(st_sz[1]))))
        # Create array for stencil values
        vof_array = np.zeros((st_sz[0], st_sz[1]))
        # Create dict to fetch shape of geometry in local coordinates
        vof_dict = {}
        if visualize:
	    # Initialize plot
            fig, (ax1, ax2) = plt.subplots(1, 2)
            # Plot circle
            plot_circle(ax1, r, x_c, x)
        # Iterate over all stencil indices combinations
        for idx, ill in enumerate(iter_loc_list):
            # Get origin coordinates too
            lo = local_origins_list[idx]
            # Values of local grid relative to geometry origin
            [y_l, x_l] = np.array([local_grid[0] + lo[0] - x_c[0],
                                   local_grid[1] + lo[1] - x_c[1]])
            # Get radii on local grid (np.multiply way faster than np.power) r^2 = x^2 + y^2 + z^2
            r_sqr = np.multiply(x_l, x_l) + np.multiply(y_l, y_l)
            # Calculate 1s and 0s on local grid 
            r_area = np.where(r_sqr <= r*r, 1, 0)
            # Get VOF values by integration over local grid
            vof = np.sum(r_area)/r_area.size
            # Write vof value into stencil value array
            vof_array[ill[0], ill[1]] = vof
            # Save the r_area array for plotting
            vof_dict[idx] = r_area
        if visualize:
            # Plot vof
            plot_vof(ax2, vof_dict, vof_array, st_sz, Delta_vof)
            # Show plot
            plt.show()
        # Only proceed if data is valid (invalid = middle point of stencil does not contain interface)
        if (vof_array[st_mid[0], st_mid[1]] > 0) & (vof_array[st_mid[0], st_mid[1]] < 1):
            # Invert values by 50% chance
            '''
            if u() > 0.5:
                curvature = -curvature
                vof_array = 1-vof_array
            '''
            # Reshape vof_array
            output_array = np.reshape(vof_array, (1, np.prod(st_sz)))[0].tolist()
            # Insert curvature value at first position
            output_array.insert(0, curvature)
            # Append list to output list
            output_list.append(output_array)
    pbar.finish()
    # Convert output list to pandas dataframe
    output_df = pd.DataFrame(output_list)
    # Reformat column names as string and rename curvature column
    output_df.columns = output_df.columns.astype(str)
    output_df = output_df.rename(columns={'0':'Curvature'})
    # Write output dataframe to feather file
    path = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(path, 'datasets', 'data'+str(st_sz[0])+'x'+str(st_sz[1])+'.feather')
    output_df.reset_index(drop=True).to_feather(file_name)
    # Print string with a summary
    print(f'Generated {output_df.shape[0]} tuples in {gt(time0)} with:\nGrid:\t\t{int(1/Delta)}x{int(1/Delta)}\nStencil size:\t{st_sz}\nVOF Grid:\t{int(1/Delta_vof)}x{int(1/Delta_vof)}\nVOF Accuracy:\t{np.round(100*Delta_vof**2,3)}%')

