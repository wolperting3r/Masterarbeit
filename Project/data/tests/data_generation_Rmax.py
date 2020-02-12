# DATA GENERATION
import numpy as np
import itertools
import time
import pandas as pd
from progressbar import *
import matplotlib.pyplot as plt
import os

from utils import (
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

def cross(mid_pt, max_pt):
    # Generate cross values mid_pt - max_pt to mid_pt + max_pt in both axis
    # Generate points in direction of both axis
    points_x = np.array([np.array([mid_pt[0]]), mid_pt[1]+max_pt[1]])
    points_y = np.array([mid_pt[0]+max_pt[0], np.array([mid_pt[1]])])
    cr_x = np.array(list(itertools.product(*points_x)))
    cr_y = np.array(list(itertools.product(*points_y)))
    # Get list of all unique cross points
    return np.unique(np.concatenate((cr_x, cr_y), axis=0), axis=0)


def generate_data(N_values, st_sz: [int, int], equal_kappa):

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
    #equal_kappa = eq_ka              # Equal kappa or equal radius
    # Stencil
    # st_sz = [5, 3]      # Stencil size y, x
    cr_sz = [3, 3]      # Cross size
    # ↑↑ Parameters ↑↑ #
    
    # Calculate midpoints
    st_mid = [int((st_sz[0]-1)/2), int((st_sz[1]-1)/2)]
    cr_mid = [int((cr_sz[0]-1)/2), int((cr_sz[1]-1)/2)]
    # Generate cross points relative to cell
    st_crp = np.array([np.arange(-cr_mid[0], (cr_mid[0]+1)),
                       np.arange(-cr_mid[1], (cr_mid[1]+1))])
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

    # Test szenarios
    R_max = [0.068, 0.069, 0.072, 0.191, 0.192, 0.2001]
    st_sz = [5, 5]
    for r in R_max:
        pbar.update(r)
        r = r
        # Calculate curvature
        curvature = L*Delta*2/r  # Stimmt das auch, wenn der Stencil anders gewählt wird?
        # Move midpoint by random amount inside one cell
        '''x_c = np.array([u(), u()])*Delta'''
        # Midpoint of VOF cell:
        x_c = np.array([15/31, 15/31, 15/31])*Delta
        # Get random spherical angle
        '''theta = 2*np.pi*u()'''
        theta = 3/2*np.pi
        # Get cartesian coordinates on sphere surface
        x_rel = np.array([r*np.sin(theta),   # y
                          r*np.cos(theta)])  # x
        x = np.array([x_c[0]+x_rel[0],
                      x_c[1]+x_rel[1]])
        # Round point to get origin of local coordinates in global coordinates relative to geometry origin
        round_point = np.floor(x*1/Delta*L)*Delta/L
            
        # > Evaluate VOF values on cross around origin first to calculate the gradient
        if st_sz[0] != st_sz[1]:
            # Create nan array for stencil values
            vof_array = np.empty((cr_sz[0], cr_sz[1]))
            vof_array[:] = np.nan
            if visualize:
                # Create dict to fetch shape of geometry in local coordinates
                vof_df = pd.DataFrame(index=range(cr_sz[0]), columns=range(cr_sz[1]))
            # Generate cross points in x and y direction
            cross_point_origins = cross(round_point, st_crp*Delta)
            cross_point_indices = cross(cr_mid, st_crp)
            for idx, ill in enumerate(cross_point_indices):
                # Get origin coordinates too
                lo = cross_point_origins[idx]
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
                if visualize:
                    # Save the r_area array for plotting
                    vof_df.iloc[ill[0], ill[1]] = r_area
            # Calculate gradient with central finite difference:
            grad_y = vof_array[cr_mid[0]+1, cr_mid[1]]-vof_array[cr_mid[0]-1, cr_mid[1]]
            grad_x = vof_array[cr_mid[0], cr_mid[1]+1]-vof_array[cr_mid[0], cr_mid[1]-1]
            normal = -1/np.sqrt(grad_y*grad_y + grad_x*grad_x)*np.array([grad_y, grad_x])
            if np.abs(normal[0]) > np.abs(normal[1]): # if gradient points more to y-direction
                # Set direction to 0 (y)
                direction = 0
                if visualize:
                    # Pad index
                    vof_df.index = vof_df.index+1
                    vof_df = vof_df.reindex(range(st_sz_loc[0]))
                # Leave stencil as it is
                st_stp_loc = st_stp
                st_sz_loc = st_sz
                st_mid_loc = st_mid
            else:
                # Set direction to 1 (x)
                direction = 1
                if visualize:
                    # Pad columns
                    vof_df.columns = vof_df.columns+1
                    vof_df = vof_df.reindex(range(st_sz_loc[1]), axis='columns').astype(object)
                # Rotate stencil by 90 degrees (flip x and y dimensions)
                st_stp_loc = np.flip(st_stp)
                st_sz_loc = np.flip(st_sz)
                st_mid_loc = np.flip(st_mid)
            # Padding of vof_array so it fits the stencil
            pad_x = int((st_sz_loc[0] - vof_array.shape[0])*1/2)
            pad_y = int((st_sz_loc[1] - vof_array.shape[1])*1/2)
            vof_array = np.pad(vof_array,
                               [[pad_x, pad_x], [pad_y, pad_y]],
                               mode='constant', constant_values=np.nan)
            if visualize:
                print(f'normal:\n{normal}')
        else:
            # If gradient is not calculated, initialize vof_array and vof_df with stencil size
            # Create nan array for stencil values
            vof_array = np.empty((st_sz[0], st_sz[1]))
            vof_array[:] = np.nan
            if visualize:
                # Create dict to fetch shape of geometry in local coordinates
                vof_df = pd.DataFrame(index=range(st_sz[0]), columns=range(st_sz[1]))
            # Leave stencil as it is
            st_stp_loc = st_stp
            st_sz_loc = st_sz
            st_mid_loc = st_mid


        if visualize:
	    # Initialize plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
            # Plot circle
            plot_circle(ax1, r, x_c, x)

        # Get origins of local coordinates of stencil points
        local_origin_points = np.array([round_point[0]+st_stp_loc[0],
                                  round_point[1]+st_stp_loc[1]])
        # Get list of all origins in stencil
        local_origins = np.array(list(itertools.product(*local_origin_points)))
        # Get list of all stencil indices combinations
        local_indices = np.array(list(itertools.product(range(st_sz_loc[0]),range(st_sz_loc[1]))))

        # Iterate over all stencil indices combinations
        for idx, ill in enumerate(local_indices):
            # Skip values that were already calculated for the gradient
            if np.isnan(vof_array[ill[0], ill[1]]):
                # Get origin coordinates too
                lo = local_origins[idx]
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
                if visualize:
                    # Save the r_area array for plotting
                    vof_df.iloc[ill[0], ill[1]] = r_area
        if visualize:
            # Print vof_array
            # print(f'vof_array:\n{vof_array}')
            # Plot vof
            plot_vof(ax2, vof_df, vof_array, st_sz_loc, Delta_vof)
            # Save figure
            path = os.path.dirname(os.path.abspath(__file__))
            file_name = os.path.join(path, 'figures', 'R_max', 'figure_' + str(r) + '.png')
            fig.tight_layout()
            fig.savefig(file_name, dpi=150)
            plt.close()
        # Only proceed if data is valid (invalid = middle point of stencil does not contain interface)
        if (vof_array[st_mid_loc[0], st_mid_loc[1]] > 0) & (vof_array[st_mid_loc[0], st_mid_loc[1]] < 1):
            # Invert values by 50% chance
            '''
            if u() > 0.5:
                curvature = -curvature
                vof_array = 1-vof_array
            '''
            # Reshape vof_array
            output_array = np.reshape(vof_array, (1, np.prod(st_sz_loc)))[0].tolist()
            # Insert curvature value at first position
            output_array.insert(0, curvature)
            # Append list to output list
            output_list.append(output_array)
    pbar.finish()
    if not visualize:
        # Convert output list to pandas dataframe
        output_df = pd.DataFrame(output_list)
        # Reformat column names as string and rename curvature column
        output_df.columns = output_df.columns.astype(str)
        output_df = output_df.rename(columns={'0':'Curvature'})
        # Write output dataframe to feather file
        path = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(path, 'datasets', 'data_'+str(st_sz[0])+'x'+str(st_sz[1])+('_eqk' if equal_kappa else '_eqr')+'.feather')
        output_df.reset_index(drop=True).to_feather(file_name)
        # Print string with a summary
        print(f'Generated {output_df.shape[0]} tuples in {gt(time0)} with:\nGrid:\t\t{int(1/Delta)}x{int(1/Delta)}\nStencil size:\t{st_sz}\nVOF Grid:\t{int(1/Delta_vof)}x{int(1/Delta_vof)}\nVOF Accuracy:\t{np.round(100*Delta_vof**2,3)}%')

