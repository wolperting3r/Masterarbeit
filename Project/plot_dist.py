import numpy as np

from src.d.data_transformation import transform_data, transform_kappa
from src.ml.building import build_model
from src.ml.training import train_model, load_model
from src.ml.validation import validate_model_loss, validate_model_plot
from src.ml.utils import param_filename
from src.d.utils import (
    u,
    pm,
)

import itertools
# import time
import pandas as pd
import matplotlib.pyplot as plt
import sys
# import math

# Disable Tensorflow output
import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def axismatrix(n_grid, axis):
    # Generate matrix where values increase along the given axis starting at 0
    n_grid = int(n_grid)
    return_matrix = np.tile(np.array(range(n_grid))/n_grid, (n_grid, 1))
    if axis == 0:
        return return_matrix.transpose((1, 0))[::-1]
    elif axis == 1:
        return return_matrix


def cross(mid_pt, max_pt, rev_y=False):
    # Generate cross values mid_pt - max_pt to mid_pt + max_pt in both axis
    # Generate points in direction of both axis

    # rev_y: reverse order in y-direction (required for cross_point_origins)
    if rev_y:
        # First take negative y values to get sorting right
        mid_pt = [-mid_pt[0], mid_pt[1]]

    # Get limits in x-direction
    points_x = np.array([np.array([mid_pt[0]]), mid_pt[1]+max_pt[1]])
    # Get limits in y-direction
    points_y = np.array([mid_pt[0]+max_pt[0], np.array([mid_pt[1]])])
    # Get points in x- and y-direction
    cr_x = np.array(list(itertools.product(*points_x)))
    cr_y = np.array(list(itertools.product(*points_y)))
    # Combine them into a list
    cross_points = np.unique(np.concatenate((cr_y, cr_x), axis=0), axis=0)

    if rev_y:
        # Now invert y values again and transpose it to original shape: np.array().transpose((1, 0))
        cross_points = np.array([-cross_points[:, 0], cross_points[:, 1]]).transpose((1, 0))

    # Get list of all unique cross points
    return cross_points


def get_vof(xs, a, f, parameters, **kwargs):
    silent = True
    L = 1
    Delta = 1/1000
    Delta_vof = 1/32

    st_sz = parameters['stencil_size']
    smearing = parameters['smear']
    if smearing:
        st_sz = np.add(st_sz, [2, 2])

    # Calculate midpoints of stencil and cross
    st_mid = [int((st_sz[0]-1)/2), int((st_sz[1]-1)/2)]
    # Generate x and y for origins of local grid for stencils points relative to local point
    st_stp = np.array([np.arange(-st_mid[0], (st_mid[0]+1)),
                       np.arange(-st_mid[1], (st_mid[1]+1))])*Delta
    # Generate local grid (y from bottom to top, x from left to right)
    local_grid = np.array([axismatrix(1/Delta_vof, 0),
                           axismatrix(1/Delta_vof, 1)])*Delta

    # Initialize list for output vectors
    output_list = []
    for x in xs:
        # Move midpoint by random amount inside one cell
        x_c = np.array([u(), u()])*Delta
        if 'xc' in kwargs:
            if not kwargs.get('xc'):
                x_c = np.array([0, 0])

        ''' Get point on geometry '''
        '''
        # Get random x 
        pt_x = 0 + u()*2/f
        # Get corresponding y
        pt_y = a*np.sin(f*np.pi*pt_x)
        # Calculate curvature 
        curvature = -2*L*Delta*(f**2*np.pi**2*pt_y)/(((a*f*np.pi)**2*(np.cos(f*np.pi*pt_x))**2+1)**(3/2))
        # '''
        # Approximate x
        pt_x = x
        # Calculate corresponding y
        pt_y = a*np.sin(f*np.pi*pt_x)
        # Calculate actual curvature of x/y
        curvature = 2*L*Delta*(f**2*np.pi**2*pt_y)/(((a*f*np.pi)**2*(np.cos(f*np.pi*pt_x))**2+1)**(3/2))

        # Rotate with random angle
        if 'rot' in kwargs:
            rot = kwargs.get('rot')
            if rot == 10:
                rot = u()*2*np.pi
        else:
            rot = u()*2*np.pi
        rot_matrix = [[np.cos(rot), -np.sin(rot)],
                      [np.sin(rot), np.cos(rot)]]
        [pt_x, pt_y] = np.matmul(rot_matrix, [pt_x, pt_y])

        # Make x array and add random shift of origin x_c
        x = np.array([pt_y, pt_x])
        x = x+x_c

        # Round point to get origin of local coordinates in global coordinates relative to geometry origin
        round_point = np.floor(x*1/Delta*L)*Delta/L

        # Initialize vof_array and vof_df with stencil size
        vof_array = np.empty((st_sz[0], st_sz[1]))
        vof_array[:] = np.nan

        # Pass on stencil values as they are
        st_stp_tmp = st_stp
        st_sz_tmp = st_sz
        st_mid_tmp = st_mid

        ''' 2. Evaluate VOF values on whole stencil '''
        # Get origins of local coordinates of stencil points
        local_origin_points = np.array([round_point[0]+st_stp_tmp[0],
                                        round_point[1]+st_stp_tmp[1]])
        # Get list of all origins in stencil
        local_origins = np.array(list(itertools.product(
            *[local_origin_points[0][::-1],  # [::-1] to get sorting right
              local_origin_points[1]]
        )))
        # Get list of all stencil indices combinations
        local_indices = np.array(list(itertools.product(range(st_sz_tmp[0]), range(st_sz_tmp[1]))))  # g

        # Iterate over all stencil indices combinations
        for idx, ill in enumerate(local_indices):
            # Skip values that were already calculated for the gradient (= not nan)
            if np.isnan(vof_array[ill[0], ill[1]]):
                # Get origin coordinates too
                lo = local_origins[idx]
                # Values of local grid relative to geometry origin (see above for note on order)
                [y_l, x_l] = np.array([local_grid[0] + lo[0] - x_c[0],
                                       local_grid[1] + lo[1] - x_c[1]])

                # Rotate geometry back to calculate geometry
                x_ltmp = x_l.copy()
                y_ltmp = y_l.copy()
                x_l = x_ltmp*np.cos(rot)+y_ltmp*np.sin(rot)
                y_l = -x_ltmp*np.sin(rot)+y_ltmp*np.cos(rot)

                # Get radii on local grid (np.multiply way faster than np.power) y = a*sin(f pi x)
                r_sqr = - y_l + a*np.sin(f*np.pi*x_l)
                # Calculate 1s and 0s on local grid
                r_area = np.where(r_sqr <= 0, 1, 0)

                # Get VOF values by integration over local grid
                vof = np.sum(r_area)/r_area.size
                # Write vof value into stencil value array
                vof_array[ill[0], ill[1]] = vof

        # Apply smearing
        if smearing:
            # Define smearing kernel
            kernel = [[0, 1, 0], [1, 4, 1], [0, 1, 0]]  # FNB
            # kernel = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]  # Gauß
            vof_array_smear = vof_array.copy()
            vof_array_smear[:] = np.nan
            for column in range(1, st_sz_tmp[0]-1):
                for row in range(1, st_sz_tmp[1]-1):
                    # Calculate smeared vof field: sum(weights * vof_array_slice)/sum(weights)
                    vof_array_smear[column, row] = np.sum(np.multiply(
                        kernel,
                        vof_array[column-1:column+2, row-1:row+2]
                    ))/np.sum(kernel)
            # Cut edges of vof_array
            vof_array = vof_array_smear[1:st_sz_tmp[0]-1, 1:st_sz_tmp[1]-1]
            # Shrink st_sz
            st_sz_tmp = np.add(st_sz_tmp, [-2, -2])
            st_mid_tmp = np.add(st_mid_tmp, [-1, -1])

        ''' Invert values by 50% chance '''
        # Only proceed if data is valid (invalid = middle point of stencil does not contain interface)
        # Invalid values are created when the interface is flat and exactly between two cells
        if (vof_array[st_mid_tmp[0], st_mid_tmp[1]] > 0) & (vof_array[st_mid_tmp[0], st_mid_tmp[1]] < 1):
            '''
                e = 100
            if True:
            # '''
            neg = False
            if neg:
                # Invert values by 50% chance
                if u() > 0.5:
                    curvature = -curvature
                    vof_array = 1-vof_array
            # Reshape vof_array from mxn array to m*nx1 vector
            output_array = np.reshape(vof_array, (1, np.prod(st_sz_tmp)))[0].tolist()
            # Insert curvature value at first position
            output_array.insert(0, curvature)
            # Append list to output list
            output_list.append(output_array)

    if smearing:
        # Shrink st_sz to create right filename
        st_sz = np.add(st_sz, [-2, -2])
    ''' Export data to feather file '''
    # Convert output list to pandas dataframe
    output_df = pd.DataFrame(output_list)
    # Reformat column names as string and rename curvature column
    output_df.columns = output_df.columns.astype(str)
    output_df = output_df.rename(columns={'0':'Curvature'})

    return output_df


def get_kappa(x, a, f, Delta, L):
    zahler = f**2*np.pi**2*a*np.sin(f*np.pi*x)
    nenner = ((a*f*np.pi)**2*(np.cos(f*np.pi*x))**2+1)**(3/2)
    kappa = np.divide(zahler, nenner)*Delta*L*2
    return kappa


def ex_plot(parameters, **kwargs):
    parameters['filename'] = param_filename(parameters)

    param_tmp = parameters.copy()
    # Set data to plotdata to load correct data file for plotting
    # param_tmp['data'] = param_tmp['plotdata']
    # param_tmp['filename'] = param_filename(param_tmp, plotdata_as_data=True)

    L = 1
    Delta = 1/1000
    Delta_vof = 1/32

    st_sz = parameters['stencil_size']
    smearing = parameters['smear']
    if smearing:
        st_sz = np.add(st_sz, [2, 2])

    f = 50
    a = 0.4/(2*L*Delta*f**2*np.pi**2)

    n_val = 1000
    x = (np.arange(0, n_val+1)/n_val)*2/f
    y = a*np.sin(f*np.pi*x)

    kappa_an = get_kappa(x, a, f, Delta, L)
    vof_field = get_vof(x, a, f, parameters, **kwargs)

    param_tmp = parameters.copy()

    plot = True

    [[train_labels, train_data, train_angle],
     [test_labels, test_data, test_angle],
     [val_labels, val_data, val_angle]] = transform_data(
         param_tmp,
         reshape=(True if parameters['network'] == 'cvn' else False),
         plot=plot,
         data=vof_field
     )  # kappa = 0 if parameters['hf'] == False
    [[train_k_labels, train_kappa], [test_k_labels, test_kappa], [val_k_labels, val_kappa]] = transform_kappa(
         param_tmp,
         reshape=(True if parameters['network'] == 'cvn' else False),
         plot=plot,
         data=vof_field
     )  # kappa = 0 if parameters['hf'] == False

    kappa_hf = test_kappa
    model = load_model(parameters)
    kappa_ml = model.predict(test_data, batch_size=parameters['batch_size']).flatten()

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    alpha = 0.9
    ax.plot(x, kappa_an, alpha=alpha, color='orange', label = 'Analytic solution')
    alpha = 0.8
    marker = 'o'
    size = 5
    ax.scatter(x, kappa_hf, alpha=alpha, color='deeppink', label = 'Height function', s=size, marker=marker, edgecolors='none')
    ax.scatter(x, kappa_ml, alpha=alpha, color='midnightblue', label = 'Machine learning', s=size, marker=marker, edgecolors='none')
    ax.xaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.set_ylabel('kappa')
    ax.set_ylim([-0.42, 0.42])
    ax.legend()
    # plt.plot(x, test_k_labels, alpha=alpha, color='blue')
    # plt.plot(x, y, alpha=alpha, color='green')

    return fig, ax

parameters = {
    'network': 'mlp',              # Network type
    'epochs': 1000,                # Number of epochs
    'layers': [100, 80],                 # Autoencoder: [n*Encoder Layers, 1*Coding Layer, 1*Feedforward Layer]
    'stencil_size': [5, 5],         # Stencil size [x, y]
    'equal_kappa': True,      # P(kappa) = const. or P(r) = const.
    'learning_rate': 1e-4,  # Learning Rate
    'batch_size': 128,        # Batch size
    'activation': 'relu',        # Activation function
    'negative': True,                 # Negative values too or only positive
    'angle': False,                  # Use the angles of the interface too
    'rotate': True,                   # Rotate the data before learning
    'data': 'all',                    # 'ellipse', 'circle', 'both'
    'smear': True,               # Use smeared data
    'hf': 'hf',                        # Use height function
    'hf_correction': False,  # Use height function as input for NN
    'plotdata': 'all',
}

rot = [10, 0.25*np.pi, 2*np.pi]
xc = [False, True]

l = [xc, rot]

combinations = list(itertools.product(*l))

for c in combinations:
    args = {'parameters': parameters, 'rot': c[0], 'xc': c[1]}
    fig, ax = ex_plot(
        **args
    )
    path = os.path.dirname(os.path.abspath(sys.argv[0]))
    file_name = os.path.join(path, 'models', 'x_plots', 'fig_' + str(parameters['stencil_size'][0]) + 'x' + str(parameters['stencil_size'][0]) + '_rot-' + ((str(c[1]/np.pi) + 'pi') if c[1] != 10 else 'rand') + '_xc-' + ('T' if c[0] else 'F') + '.png')
    fig.tight_layout()
    fig.savefig(file_name, dpi=150)
