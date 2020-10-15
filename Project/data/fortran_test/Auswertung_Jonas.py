import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import tikzplotlib as tkz
import regex
import re

# Initialize figure and axis
fig, ax = plt.subplots(1, 1, figsize=(10,5))

# '''
# Folders with y_pos_re.txt (generated with update.py)
paths = [
    'FASTEST_1',
    'FASTEST_2',
    'FASTEST_3',
    'FASTEST_4',
    '2006031403 CVOFLS',
    '2006022000 cds Vergleich',
]
# Labels for graphs
labels = [
    'F1',
    'F2',
    'F3',
    'F4',
    'CVOFLS',
    'CDS Vergleich',
]
#'''

# Define colors
reds = ['tomato', 'red', 'maroon', 'brown']
# Counters for colors
j=0
k=0
# Plot until endtime
endtime = 2 # in s
# Timestep of simulation
timestep = 0.000625

# Iterate through colors
colors=iter(plt.cm.rainbow(np.linspace(0,1,len(paths))))
for i in range(len(paths)):
    # Get path to y_pos_re.txt (y_pos.txt processed with regex)
    path = paths[i]
    # Get corresponding label
    label = labels[i]
    # Get path to file
    filepath = os.path.join(path, 'y_pos_re.txt')
    # Try to read file
    if os.path.isfile(filepath):
        data = pd.read_csv(filepath, skiprows=1, names=['it', 'x', 'y', 'c', 'cm'])
    else:
        print(f'file {path} not found!')

    # Number of values per timestep
    diffpos = data['it'].value_counts().iloc[0]

    # Cut data at endtime
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

    if (label == 'CVOFLS') or ('MSE' in label):
        alpha = 1
    else:
        alpha = 0.5
    linewidth = 1.5

    if (re.match(r'F\d', label) or ('altes' in label)):
        # Plot parameters for fastest plots
        color = reds[j]
        # color = 'gold'
        j = j+1
        linewidth = 2
    elif ('cds' in label) or ('CDS' in label):
        color = 'chocolate'
    elif (('CVOFLS' in label)):
        color = 'lime'
    else:
        color = next(colors)

    # Plot linear interpolated y position
    pd.Series(y_05).plot(label=label, c=color, alpha=alpha, linewidth=linewidth)


# Set x-ticks to analytical period
ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: f'{x*timestep}'))
plt.xticks(np.arange(start=0, step=1.5888/(2*timestep), stop=int((endtime))/timestep))

# Set y-ticks
ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos: '%.2E' % np.round((y-0.075/2)*0.1, 4)))
plt.yticks(np.arange(start=(0.002*10)+0.075/2, step=0.005 , stop=(0.003*10)+0.075/2))
# Set axis labels 
ax.set_ylabel('y-Pos [m]')
ax.set_xlabel('Zeit [s]')
# ax.set_ylim([0.0540,0.0661])

# Plot vertical lines at n*1/2*analytical period
for i in range(int(np.floor(2*endtime/1.5888)+1)):
    plt.axvline(x=i/2*1.5888/timestep, color='k', lw=0.5)

# Make layout tight
fig.tight_layout()
# Plot legend
plt.legend()
# Save figure
plt.savefig('result.png', dpi=150)
# Show figure
plt.show()
