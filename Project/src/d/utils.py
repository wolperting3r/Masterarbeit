import numpy as np
import matplotlib.pyplot as plt
import time

def gt(time0):
    return str(f'{np.round(time.time() - time0,3)} s')


def plot_circle(ax1, r, x_c, x):
    ''' Plot circle with chosen point '''
    # Number of values for circle edge
    n_val = 250
    # Generate evenly distributed values around circle
    values = np.arange(0, n_val+1)/n_val
    theta_plt = 2*np.pi*values
    # Get x and y values of circle edge
    # !! x_c = [-x_c_y, x_c_x]
    [x_plt, y_plt] = [r*np.cos(theta_plt)+x_c[1], r*np.sin(theta_plt)-x_c[0]]
    # Plot circle
    ax1.fill(x_plt, y_plt, color='w', zorder=-1)
    # Plot point [-y, x]
    ax1.scatter(x[1], -x[0], color='r')
    ax1.set_facecolor('k')
    # Print radius 
    ax1.text(0.05, 0.05, f'r = {np.round(r,3)}', transform=ax1.transAxes, color='w')
    # Make axis equally long
    ax1.set_xlim([-0.5, 0.5])
    ax1.set_ylim([-0.5, 0.5])
    ax1.set_aspect('equal')


def plot_ellipse(ax1, r, e, x, rot, curvature):
    # Number of values for circle edge
    n_val = 250
    # Generate evenly distributed values around circle
    values = np.arange(0, n_val+1)/n_val
    theta_plt = 2*np.pi*values
    # Get x and y values of circle edge
    [x_plt, y_plt] = [r/e*np.cos(theta_plt), r*np.sin(theta_plt)]
    x_plt_tmp = x_plt.copy()
    y_plt_tmp = y_plt.copy()
    x_plt = x_plt_tmp*np.cos(-rot) - y_plt_tmp*np.sin(-rot)
    y_plt = x_plt_tmp*np.sin(-rot) + y_plt_tmp*np.cos(-rot)
    # Plot circle
    ax1.fill(x_plt, y_plt, color='w', zorder=-1)
    # Plot point [-y, x]
    ax1.scatter(x[1], -x[0], color='r')
    ax1.set_facecolor('k')
    # Print radius 
    ax1.text(0.5, 0.5, f'r = {np.round(r,3)}\nk = {np.round(curvature,3)}', transform=ax1.transAxes, color='k', ha='center')
    # Make axis equally long
    # ax1.set_xlim([-0.5, 0.5])
    # ax1.set_ylim([-0.5, 0.5])
    ax1.set_aspect('equal')



def plot_vof(ax2, vof_df, vof_array, st_sz, Delta_vof):
    ''' Plot stencil with geometry and vof values '''
    # print(f'vof_df:\n{vof_df}')
    # Initialize image array
    image = np.array([])
    for column in range(st_sz[0]):  # y
        # Initialize row array
        column_values = np.array([])
        # Glue arrays in vof_df together
        for row in range(st_sz[1]):  # x
            if len(column_values) == 0:
                # row_values = vof_df[row*st_sz[1]+column]
                # row_values = vof_df.iloc[row, column]
                column_values = vof_df.iloc[row, column]
            else:
                # row_values = np.concatenate((row_values, vof_df[row*st_sz[1]+column]), axis=1)
                # row_values = np.concatenate((row_values, vof_df.iloc[row, column]), axis=1)
                column_values = np.concatenate((column_values, vof_df.iloc[row, column]), axis=1)
        # Glue rows together
        if len(image) == 0:
            image = column_values
        else:
            image = np.concatenate((image, column_values), axis=0)
    # Get image dimensions to calculate grid
    imgdim = np.array([st_sz[0], st_sz[1]])*1/Delta_vof
    # Set ticks
    x_ticks = np.arange(0, imgdim[1]+1, int(1/Delta_vof))
    y_ticks = np.arange(0, imgdim[0]+1, int(1/Delta_vof))
    ax2.set_xticks(x_ticks)
    ax2.set_yticks(y_ticks)
    # Set grid
    ax2.grid(which='both')
    # Generate vof labels
    vs = vof_array.shape
    for row in range(vs[0]):
        for column in range(vs[1]):
            # Calculate position of text
            txt_pt = [column*1/Delta_vof+(1/Delta_vof)/10, row*1/Delta_vof+(1/Delta_vof)/10]
            # Write vof value at that position
            ax2.text(txt_pt[0], txt_pt[1],
                 np.round(vof_array[column, row], 3),
                 horizontalalignment='left',
                 verticalalignment='top',
                 color='k',
                 backgroundcolor='w')
    # Show geometry
    ax2.imshow(image, cmap='Greys_r')

