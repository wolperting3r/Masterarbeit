def plotsphere(r, x_c):
    n_points = 100
    plot_values = np.array(range(0,n_points))/n_points
    # plot_angles = [np.arccos(2*plot_values-1), 2*np.pi*plot_values]
    plot_angles = [np.pi*plot_values, 2*np.pi*plot_values]
    plot_angles = np.array(list(itertools.product(*plot_angles)))
    print(f'plot_angles: {plot_angles}')
    print(f'plot_angles.shape: {plot_angles.shape}')
    x_plot = np.array([r*np.cos(plot_angles[:, 0])*np.sin(plot_angles[:, 1]),
                       r*np.sin(plot_angles[:, 0])*np.sin(plot_angles[:, 1]),
                       r*np.cos(plot_angles[:, 1])])
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x_plot[0], x_plot[1], x_plot[2])
    '''
    x_plot = [r*np.outer(np.cos(plot_angles[:, 0]), np.sin(plot_angles[:, 1])),
              r*np.outer(np.sin(plot_angles[:, 0]), np.sin(plot_angles[:, 1])),
              r*np.outer(np.ones(np.size(plot_angles[:,0])), np.cos(plot_angles[:, 1]))]
    print(f'x_plot: {x_plot}')
    print(f'plot_angles[:,0]: {plot_angles[:,0]}')
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    ax = Axes3D(fig)
    ax.plot_wireframe(x_plot[0], x_plot[1], x_plot[2], alpha=0.5)
    elev = 10.0
    rot = 80.0 / 180 * np.pi

    #ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', linewidth=0, alpha=0.5)

    #calculate vectors for "vertical" circle
    a = np.array([-np.sin(elev / 180 * np.pi), 0, np.cos(elev / 180 * np.pi)])
    b = np.array([0, 1, 0])
    b = b * np.cos(rot) + np.cross(a, b) * np.sin(rot) + a * np.dot(a, b) * (1 - np.cos(rot))
    ax.plot(r*np.sin(plot_angles[:, 0]),r*np.cos(plot_angles[:, 0]),0,color='k', linestyle = 'dashed')
    horiz_front = np.linspace(0, np.pi, 100)
    ax.plot(r*np.sin(horiz_front),r*np.cos(horiz_front),0,color='k')
    vert_front = np.linspace(np.pi / 2, 3 * np.pi / 2, 100)
    ax.plot(a[0] * r*np.sin(plot_angles[:, 0]) + b[0] * r*np.cos(plot_angles[:, 0]), b[1] * r*np.cos(plot_angles[:, 0]), a[2] * r*np.sin(plot_angles[:, 0]) + b[2] * r*np.cos(plot_angles[:, 0]),color='k', linestyle = 'dashed')
    ax.plot(a[0] * r*np.sin(vert_front) + b[0] * r*np.cos(vert_front), b[1] * r*np.cos(vert_front), a[2] * r*np.sin(vert_front) + b[2] * r*np.cos(vert_front),color='k')

    ax.scatter([x_c[0]], [x_c[1]], [x_c[2]], color='r')

    ax.view_init(elev = elev, azim = 0)
    '''

    plt.show()
