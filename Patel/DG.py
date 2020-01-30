# DATA GENERATION
import numpy as np
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
    # '''
    #-- Parameters --#
    # Script
    N_values = 10           # Number of values
    # Grid
    Delta = 1/1000          # Gridsize
    Delta_vof = 1/100      # VoF Gridsize
    L = 1                   # Length of space
    X_c = np.array([L/2, L/2, L/2])   # Midpoint of geometry
    # Geometry
    R_min = 0.0002         # Minimal radius (TBD)
    R_max = 0.5            # Maximal radius
    # Stencil
    st_sz = [3, 3, 3]      # Stencil size
    st_stp = np.array([np.arange(-(st_sz[0]-1)/2, (st_sz[0]-1)/2+1), np.arange(-(st_sz[1]-1)/2, (st_sz[1]-1)/2+1), np.arange(-(st_sz[1]-1)/2, (st_sz[1]-1)/2+1)])*Delta    # Stencil points relative to current cell
    # Calulate local grid
    local_grid = [axismatrix(1/Delta_vof, 0), axismatrix(1/Delta_vof, 1), axismatrix(1/Delta_vof, 2)]
    for n in range(N_values):
        # Get random radius
        r = R_min + u()*(R_max - R_min)
        # Move midpoint by random amount inside one cell
        x_c = X_c + np.array([u(), u(), u()])*Delta
        # Get random spherical angles
        theta = 2*np.pi*u()
        psi = np.arccos(2*u()-1)
        # Get cartesian coordinates on sphere surface
        x_rel = np.array([r*np.cos(theta)*np.sin(psi), r*np.sin(theta)*np.sin(psi), r*np.cos(psi)])
        x_abs = np.array([x_c[0]+x_rel[0], x_c[1]+x_rel[1], x_c[2]+x_rel[2]])
        # Round point
        round_point = np.floor(x_rel*1/Delta*L)*Delta/L
        start_points = [round_point[0]+st_stp[0], round_point[1]+st_stp[1], round_point[2]+st_stp[2]]
        # Das hier muss für jeden Punkt in allen Kombinationen von start_points geschehen
        # Values of local grid in absolute values
        [x_l, y_l, z_l] = [local_grid[0] + round_point[0], local_grid[1] + round_point[1], local_grid[2] + round_point[2]]
        # Und dann die VoF Werte bestimmen und auswerten (np.where( >r), summe bilden)

        # Das hier brauche ich vielleicht gar nicht
        # Get point on grid
        indices = np.floor((x_abs+X_c*Delta)*1/Delta*L).astype(int)

    # '''
