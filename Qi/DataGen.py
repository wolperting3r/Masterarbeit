import numpy as np
import pandas as pd
import os
# Declare variables
# npp = 256000
npp = 4
npp = npp + 1  # Damit konsisten mit Matlab Skript
nx = 2000
L = 1

DTheta = 2*np.pi/(npp-2)  # Matlab hat hier -1, d.h. um auf gleiche Werte bei gleich vielen Werten zu kommen muss hier -2 rein.
xo = 0.5
yo = 0.5

stepwidth1 = 0.001
stepwidth2 = 0.025
r1 = np.arange(0.00225, 0.05 + stepwidth1, stepwidth1)
r2 = np.arange(0.075, 0.475 + stepwidth2, stepwidth2)
r = np.concatenate((r1, r2))
r = [0.00225]
datacur = []
datafra = []

for ir in r:
    ro = ir
    xp = []
    yp = []
    # Constructing the interface shape
    """
    for i in range(1,npp):
        theta = DTheta*(i-1)
        # x = x0 + r * cos(theta), y = y0 + r * sin(theta)
        xp.append(xo+ro*np.cos(theta))
        yp.append(yo+ro*np.sin(theta))
    """
    i = np.arange(0, npp, 1)
    theta = DTheta*i
    xp = xo+ro*np.cos(theta)
    yp = yo+ro*np.sin(theta)

    print(f'xp: {xp}')
    print(f'yp: {yp}')

    # Construct fixed grid
    ny = nx
    # Gridsize = Height/n_x
    h = L/(nx-1)
    # Get grid values
    """
    for i in range(1,nx):
       x.append(h*(i-1))
    for j in range(1,ny):
       y.append(h*(j-1))
    """
    i = np.arange(0, nx, 1)
    j = np.arange(0, ny, 1)
    x = h*i
    y = h*j

    print(f'x: {np.round(x,4)}')
    print(f'y: {np.round(y,4)}')
    print(f'shape x: {len(x)}')
    print(f'shape y: {len(y)}')

    # Construct the marker
    # xi = np.zeros((nx-1, ny-1))
    xi = np.zeros((nx, ny))

    for i in range(npp):
        if i % 1000 == 0:
            print(i)
        print(f'i: {i}')
        # ip1 = np.floor(xp[i-1]*(nx-1))+1
        # ip2 = np.floor(xp[i]*(nx-1))+1

        # jp1 = np.floor(yp[i-1]*(ny-1))+1
        # jp2 = np.floor(yp[i]*(ny-1))+1

        # Get coordinates of point and next point
        ip1 = np.floor(xp[i]*(nx-1))
        ip2 = np.floor(xp[i+1]*(nx-1))

        jp1 = np.floor(yp[i]*(ny-1))
        jp2 = np.floor(yp[i+1]*(ny-1))

        # Add two in-between points
        xs = [xp[i], xp[i+1], xp[i+1], xp[i+1]]
        ys = [yp[i], yp[i+1], yp[i+1], yp[i+1]]

        if (ip1 != ip2):
            if (ip1 < ip2):
                xv = (ip2-1)*h
            elif (ip1 > ip2):
                xv = (ip1-1)*h
            yv = yp[i-1]+((yp[i]-yp[i-1])/(xp[i]-xp[i-1]))*(xv-xp[i-1])
            # yv = yp[i]+((yp[i+1]-yp[i])/(xp[i+1]-xp[i]))*(xv-xp[i])

        if (jp1 != jp2):
            if (jp1 < jp2):
                yh = (jp2-1)*h
            elif (jp1 > jp2):
                yh = (jp1-1)*h
            xh = xp[i-1]+((xp[i]-xp[i-1])/(yp[i]-yp[i-1]))*(yh-yp[i-1])
            # xh = xp[i]+((xp[i+1]-xp[i])/(yp[i+1]-yp[i]))*(yh-yp[i])

        if ((ip1 < ip2) & (jp1 == jp2)):
            xs[2] = xv
            ys[2] = yv
        if ((jp1 != jp2) & (ip1 == ip2)):
            xs[2] = xh
            ys[2] = yh

        if ((ip1 < ip2) & (jp1 != jp2)):
            xs[2] = xv
            ys[2] = yv
            xs[3] = xh
            ys[3] = yh
            if (xv > xh):
                xs[2] = xh
                ys[2] = yh
                xs[3] = xv
                ys[3] = yv

        if ((ip1 > ip2) & (jp1 != jp2)):
            xs[2] = xv
            ys[2] = yv
            xs[3] = xh
            ys[3] = yh
            if (xv < xh):
                xs[2] = xh
                ys[2] = yh
                xs[3] = xv
                ys[3] = yv

        for j in range(3):
            print(j)
            # ip = int(np.floor(0.5*(xs[j-1]+xs[j])*(nx-1))+1)
            ip = int(np.floor(0.5*(xs[j]+xs[j+1])*(nx-1)))
            # jp = int(np.floor(0.5*(ys[j-1]+ys[j])*(ny-1))+1)
            jp = int(np.floor(0.5*(ys[j]+ys[j+1])*(ny-1)))
            # dx = -(xs[j]-xs[j-1])
            dx = -(xs[j+1]-xs[j])
            # Xi ist wohl der stencil
            xi[ip, jp] = xi[ip, jp]+(0.5*(ys[j]+ys[j+1])-y[jp])*dx/h/h
            for k in range(jp-1):
                xi[ip, k] = xi[ip, k]+h*dx/h/h

    ip0 = -1
    jp0 = -1
    ncells = 0
    cellloc = np.zeros((npp, 2))
    cur = np.zeros((npp, 1))
    fra = np.zeros((npp, 9))
    for i in range(npp-1):
        ip = np.floor(xp[i]*(nx-1))+1
        jp = np.floor(yp[i]*(ny-1))+1
        if ((ip != ip0) | (jp != jp0)):
            ncells = ncells+1
            ip0 = ip
            jp0 = jp
            cellloc[ncells, :] = [ip0, jp0]
    if all(cellloc[ncells, :] == cellloc[1, :]):
        ncells = ncells-1

    for i in range(ncells):
        cur[i] = h/ro
        fra[i, 0] = xi[int(cellloc[i, 0]-1), int(cellloc[i, 1]-1)]
        fra[i, 1] = xi[int(cellloc[i, 0]-1), int(cellloc[i, 1])]
        fra[i, 2] = xi[int(cellloc[i, 0]-1), int(cellloc[i, 1]+1)]
        fra[i, 3] = xi[int(cellloc[i, 0]), int(cellloc[i, 1]-1)]
        fra[i, 4] = xi[int(cellloc[i, 0]), int(cellloc[i, 1])]
        fra[i, 5] = xi[int(cellloc[i, 0]), int(cellloc[i, 1]+1)]
        fra[i, 6] = xi[int(cellloc[i, 0]+1), int(cellloc[i, 1]-1)]
        fra[i, 7] = xi[int(cellloc[i, 0]+1), int(cellloc[i, 1])]
        fra[i, 8] = xi[int(cellloc[i, 0]+1), int(cellloc[i, 1]+1)]
    cur = cur[range(ncells), :]
    fra = fra[range(ncells), :]

    datacur.append(cur)
    datafra.append(fra)


datacur = np.round(datacur[0], 5)
datafra = np.round(datafra[0], 5)

# datacur.append(-datacur)
# datafra.append(-datafra)

datacur = pd.DataFrame(datacur)
datafra = pd.DataFrame(datafra)

path = os.path.dirname(os.path.abspath(__file__))
datacur.to_csv(os.path.join(path, 'datacur.csv'), index=False)
datafra.to_csv(os.path.join(path, 'datafra.csv'), index=False)
