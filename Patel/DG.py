# DATA GENERATION
import numpy as np
import itertools
import time
import pandas as pd
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
def gt(time0):
    return str(f'{np.round(time.time() - time0,3)} s')


def u():
    return np.random.uniform()


def axismatrix(n_grid, axis):
    # Generate matrix where values increase along the given axis starting at 0
    n_grid = int(n_grid)
    # Ascending values in z-direction
    return_matrix = np.tile(np.array(range(n_grid))/n_grid, (n_grid, n_grid, 1))
    if axis == 0:
        return return_matrix.transpose((2, 1, 0))
    elif axis == 1:
        return return_matrix.transpose((1, 2, 0))
    elif axis == 2:
        return return_matrix


if __name__ == '__main__':
    time0 = time.time()

    # ↓↓ Parameters ↓↓ #
    # Script
    N_values = 10000        # Number of values
    # Grid
    Delta = 1/1000          # Gridsize
    Delta_vof = 1/10        # VoF Gridsize 0.1% accuracy: 32 -> ~1000 (1024) points
    L = 1                   # Length of space
    X_c = np.array([L/2, L/2, L/2])   # Midpoint of geometry
    # Geometry
    R_min = 0.0002         # Minimal radius (TBD)
    R_max = 0.5            # Maximal radius
    # Stencil
    st_sz = [3, 3, 3]      # Stencil size 
    # ↑↑ Parameters ↑↑ #
    
    # Generate stencil points relative to cell
    st_stp = np.array([np.arange(-(st_sz[0]-1)/2, (st_sz[0]-1)/2+1),
                       np.arange(-(st_sz[1]-1)/2, (st_sz[1]-1)/2+1),
                       np.arange(-(st_sz[2]-1)/2, (st_sz[2]-1)/2+1)])*Delta
    # Generate a local grid for later usage
    local_grid = np.array([axismatrix(1/Delta_vof, 0),
                           axismatrix(1/Delta_vof, 1),
                           axismatrix(1/Delta_vof, 2)])*Delta
    # Initialize output list
    output_list = []
    for n in range(N_values):
        # Get random radius
        r = R_min + u()*(R_max - R_min)
        # Move midpoint by random amount inside one cell
        x_c = np.array([u(), u(), u()])*Delta
        # Get random spherical angles
        theta = 2*np.pi*u()
        psi = np.arccos(2*u()-1)
        # Get cartesian coordinates on sphere surface
        x_rel = np.array([r*np.cos(theta)*np.sin(psi),
                          r*np.sin(theta)*np.sin(psi),
                          r*np.cos(psi)])
        x_rel = np.array([x_c[0]+x_rel[0],
                          x_c[1]+x_rel[1],
                          x_c[2]+x_rel[2]])
        # Round point to get origin of local coordinates in global coordinates relative to geometry origin
        round_point = np.round(np.floor(x_rel*1/Delta*L)*Delta/L,3)
        # Get origins of local coordinates of stencil points
        local_origins = np.array([round_point[0]+st_stp[0],
                                 round_point[1]+st_stp[1],
                                 round_point[2]+st_stp[2]])
        # Get list of all origins in stencil
        local_origins_list = np.array(list(itertools.product(*local_origins)))
        # Get list of all stencil indices combinations
        iter_loc_list = np.array(list(itertools.product(range(st_sz[0]),range(st_sz[1]),range(st_sz[2]))))
        # Create array for stencil values
        vof_array = np.zeros((st_sz[0], st_sz[1], st_sz[2]))
        # Iterate over all stencil indices combinations
        for idx, ill in enumerate(iter_loc_list):
            # Get origin coordinates too
            lo = local_origins_list[idx]
            # Values of local grid relative to geometry origin
            [x_l, y_l, z_l] = np.array([local_grid[0] + lo[0],
                               local_grid[1] + lo[1],
                               local_grid[2] + lo[2]])
            # Get radii on local grid (np.multiply way faster than np.power) r^2 = x^2 + y^2 + z^2
            r_sqr = np.multiply(x_l, x_l) + np.multiply(y_l, y_l) + np.multiply(z_l, z_l)
            # Calculate 1s and 0s on local grid 
            r_area = np.where(r_sqr <= r*r, 1, 0)
            # Get VOF values by integration over local grid
            vof = np.round(np.sum(r_area)/r_area.size, 3)
            # Write vof value into stencil value array
            vof_array[ill[0], ill[1], ill[2]] = vof
        # Calculate curvature
        curvature = L*Delta*2/r  # Stimmt das auch, wenn der Stencil anders gewählt wird?
        # Invert values by 50% chance
        if u() > 0.5:
            curvature = -curvature
            vof_array = 1-vof_array
        # Reshape vof_array
        output_array = np.reshape(vof_array, (1, np.prod(st_sz)))[0].tolist()
        # Insert curvature value at first position
        output_array.insert(0, curvature)
        # Append list to output list
        output_list.append(output_array)
    # Convert output list to pandas dataframe
    output_df = pd.DataFrame(output_list)
    # Reformat column names as string and rename curvature column
    output_df.columns = output_df.columns.astype(str)
    output_df = output_df.rename(columns={'0':'Curvature'})
    # Write output dataframe to feather file
    output_df.reset_index(drop=True).to_feather('data.feather')
    # Print string with a summary
    print(f'Generated {N_values} tuples in {gt(time0)} with:\nGrid:\t\t{int(1/Delta)}x{int(1/Delta)}\nStencil size:\t{st_sz}\nVOF Grid:\t{int(1/Delta_vof)}x{int(1/Delta_vof)}\nVOF Accuracy:\t{np.round(100*Delta_vof**2,3)}%')
