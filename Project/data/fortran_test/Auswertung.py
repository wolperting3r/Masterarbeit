import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

# path = './2006010847 Stabiles Modell dshift1'
# path = './2006010848 Stabiles Modell shift1'
path = './2006011726 Stabiles Modell dshift1b'
paths = ['./2006011726 Stabiles Modell dshift1b', './2006010848 Stabiles Modell shift1', './2006010847 Stabiles Modell dshift1']
labels = ['dshift1b', 'shift1', 'dshift1']
for i in range(3):
    path = paths[i]
    label = labels[i]
    data = pd.read_csv(os.path.join(path, 'y_pos_re.txt'), skiprows=1, names=['it', 'x', 'y', 'c', 'cm'])
    endtime = 10  # in s
    timestep = 0.000625
    diffpos = 0
    diffpos = data['it'].value_counts().iloc[0]

    # Cut 10 s
    data = data.iloc[:np.int32(diffpos*endtime*1/timestep)+diffpos, :]


    # Convert to numpy array
    data = data.values

    # Reshape cm into 2D-array
    c = np.reshape(data[:, 4], (int(data.shape[0]/diffpos), diffpos))
    # Cut off below and above threshold
    tolerance = 1e-2
    c[np.nonzero(c < tolerance)] = 0
    c[np.nonzero(c > 1-tolerance)] = 1

    # Reshape y into 2D-array
    y = np.reshape(data[:, 2], (int(data.shape[0]/diffpos), diffpos))

    # Find closest value to c=0.5 where c>0.5
    indices = np.argmin(np.where((c-0.5) < 0, c.max(), c), axis=1)
    c_upper = c[range(c.shape[0]), indices]
    y_upper = y[range(y.shape[0]), indices]

    # Find closest value to c=0.5 where c<0.5
    indices = np.argmax(np.where(-(c-0.5) <= 0, 0, c), axis=1)
    c_lower = c[range(c.shape[0]), indices]
    y_lower = y[range(y.shape[0]), indices]

    # Make array with right dimensions filled with 0.5
    c_mid = c_lower.copy()
    c_mid[:] = 0.5

    # Get y where c = 0.5 by linear interpolation
    y_05 = y_lower + (c_mid-c_lower)/(c_upper-c_lower)*(y_upper-y_lower)


    pd.Series(y_05).plot(label=label)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: f'{x*0.000625}'))
    ax.set_ylabel('y-pos')
    ax.set_xlabel('time [s]')
plt.legend()
plt.show()
