import numpy as np
import time
import itertools
import pandas as pd
# Generate circle on grid
# Get volume fractions on grid
# Get curvature for cells that contain the edge
# Get VF values around that cells
# Enable full width output for numpy (https://stackoverflow.com/questions/43514106/python-terminal-output-width)
np.set_printoptions(suppress=True,linewidth=250,threshold=250)

def gt(time0):
	'''
	Return time passed since reference time time0 (for debugging purposes)
	---
	Input
	time0[time]: reference time
	---
	Output
	string: String with time since reference time + ' s'
	'''
	return str(f'{np.round(time.time() - time0,3)} s')

def axisMatrix(n_grid, axis):
    # Generate matrix where values increase along the given axis starting at 0
    return_matrix = np.ones((n_grid, n_grid))
    return_matrix = np.add(np.cumsum(return_matrix, axis=axis), -1)
    '''
    # Flip matrix so values increase from bottom to top for y-matrix
    if axis == 0:
        return_matrix = np.flip(return_matrix, axis=axis)
    '''
    return return_matrix

if __name__ == '__main__':
    # Get time for performance measurement
    time0 = time.time()

    ### PARAMETERS ###
    # > Set parameters for grid creation
    delta_s = 2000    # Number of points-1 in one axis for small grid (should be an odd number so the midpoint of geometry lies on a gridpoint
    delta_b = 100    # Number of points per axis per cell (around one gridpoint of small grid) for big grid
    height = 1       # Height of small grid
    # Stencil size (sz_x x sz_y) (only odd numbers valid):
    [sz_x, sz_y] = [3, 3]
    ### PARAMETERS ###

    # Calculate stencil step
    [szs_x, szs_y] = [int((sz_x-1)/2), int((sz_y-1)/2)]
    # Initialize small grid (2001x2001 points)
    [x_s, y_s] = [axisMatrix(delta_s+1, 1), axisMatrix(delta_s+1, 0)]
    # Midpoint of geometry
    [x0, y0] = [height/2*delta_s, height/2*delta_s]

    # Create radii from 0.00225 to 0.05 with stepwidth 0.001 and from 0.05 to 0.475 with stepwidth 0.025
    stepwidth1 = 0.001
    stepwidth2 = 0.025
    radii_1 = np.arange(0.00225, 0.05, stepwidth1)  # Ursprünglich bei 0.00225 beginnend
    radii_2 = np.arange(0.075, 0.475 + stepwidth2, stepwidth2)
    radii = np.round(np.concatenate((radii_1, radii_2)),5)
    # Ininitalize output list
    output_list = []
    # Iterate over radii
    for r in radii:
        # print(f'Radius: {r}')
        # Get geometry on small grid
        r_mat = np.sqrt(np.power(x_s-x0, 2) + np.power(y_s-y0, 2))
        r_area = np.where(r_mat <= r*delta_s, 1, 0)
        # > Get points around edge of geometry on small grid (the big grid will only be calculated on those points)
        # Find points around the edge by calculating the difference in all directions (up, down, left, right), summing the absolute values of all 4 arrays, setting it to 1 where the result is > 0 and to 0 everywhere else and getting the indices of all points that are 1.
        diff = [(np.abs(np.diff(r_area, axis=i, append=0)) if j == 0 else np.abs(np.diff(r_area, axis=i, prepend=0))) for i in [0, 1] for j in [0, 1]]  # list of 4 arrays containing 0s and positive values
        # Get indices of points where big grid should be calculated (= where sum of all arrays is > 0)
        arg = np.argwhere(np.where(np.sum(diff, axis=0) > 0, 1, 0))
        # > Create big grid around those points and get volume of fluid
        vof_grid = r_area.astype(float)
        for a in arg:
            # Global coordinates of cell center
            [x_mid, y_mid] = [a[1], a[0]]
            # print(f'x_mid, y_mid: {[x_mid, y_mid]}')
            # Get origin of cell in global coordinates
            [x0_cg, y0_cg] = [x_mid - 0.5, y_mid - 0.5]
            # Get local coordinates
            [x_cl, y_cl] = [axisMatrix(delta_b, 1)*1/delta_b, axisMatrix(delta_b, 0)*1/delta_b]
            # Transform to global coordinates
            [x_cg, y_cg] = [x_cl + x0_cg, y_cl + y0_cg]
            # Get section of geometry
            r_area_c = np.sqrt(np.power(x_cg - x0, 2) + np.power(y_cg - y0, 2))
            r_area_c = np.where(r_area_c <= r*delta_s, 1, 0)
            # Calculate VoF
            vof = np.sum(r_area_c)/r_area_c.size
            vof_grid[y_mid, x_mid] = vof
        # Pad array with zeros so stencil points do not exceed array index
        vof_grid = np.pad(vof_grid, ((szs_y, szs_y), (szs_x, szs_x)), 'constant', constant_values = (0))
        # Print vof grid
        # print(f'vof_grid:\n{vof_grid}')
        # Get coordinates of cells that lie on geometry edge
        r_edge = np.where((vof_grid > 0) & (vof_grid < 1), 1, 0)
        edge_points = np.argwhere(r_edge)
        print(f'Radius: {r},    Number of points: {len(edge_points)}')
        for p in edge_points:
            # Midpoint of stencil:
            [x_p, y_p] = [p[1], p[0]]
            # Get stencil values
            stencil_values = vof_grid[int(p[0]-szs_y):int(p[0]+szs_y+1), int(p[1]-szs_x):int(p[1]+szs_x+1)]
            # Reshape sz_x x sz_y array to 1 x sz_x*sz_y array
            stencil_values = np.reshape(stencil_values, (1, sz_x*sz_y)).tolist()[0]
            # Get curvature (hard coded for circle)
            # curvature = 1/r
            curvature = height/(delta_s-1)*1/r
            # Insert curvature into list
            stencil_values.insert(0, curvature)
            # Append list with values to output list
            output_list.append(stencil_values)
    # Convert output_list to pandas dataframe
    output_list = pd.DataFrame(output_list)
    # Reformat column names as string and rename the curvature column
    output_list.columns = output_list.columns.astype(str)
    output_list = output_list.rename(columns={'0':'Curvature'})
    # > Generate inverse values
    output_list_inverse = output_list.copy(deep=True)
    # Inversed curvature = -curvature
    output_list_inverse.iloc[:, 0] = -output_list_inverse.iloc[:, 0]
    # Inversed vof values = 1-vof values
    output_list_inverse.iloc[:, 1:] = 1-output_list_inverse.iloc[:, 1:]
    # Glue the two dataframes together
    output_list = output_list.append(output_list_inverse).reset_index(drop=True)
    # Print output list
    # print(f'output_list:\n{output_list}')
    # Save output list dataframe to feather file
    output_list.to_feather('data.feather')
    # Print end string
    print(f'Finished generation of {len(radii)} radii on {delta_s}x{delta_s} grid with {delta_b}x{delta_b} subgrid and {sz_x}x{sz_y} stencil in {gt(time0)}.')
