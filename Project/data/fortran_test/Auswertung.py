import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import re

fig, ax = plt.subplots(1, 1, figsize=(10,8))
# path = './2006010847 Stabiles Modell dshift1'
# path = './2006010848 Stabiles Modell shift1'
# path = './2006011726 Stabiles Modell dshift1b'
# path = './2006020903 cds Vergleich'
# path = './2006020903 shift1 dshift1b Artefakte'
# paths = ['2006031403 CVOFLS', '2006022000 cds Vergleich', '2006022001 altes curv_ml', 'FASTEST_3', 'FASTEST_4']
# labels = ['Referenz 2 (CVOFLS FNB)', 'CDS', 'ML', 'F3', 'F4']

# paths = ['2006040836 altes w bei k ungleich 0 0.08 0.92 wenig Fehler kleine Amplitude', '2006040837 altes g bei k ungleich 0 0.08 0.92 wenig Fehler kleine Amplitude', '2006040838 altes g 0.01 0.99 Fehler führen zu großer Amplitude', '2006040839 altes 0.01 0.98 0.005 0.995', '2006031403 CVOFLS', '2006031405 shift1','2006022001 altes curv_ml']
# labels = ['altes w 0.08 0.92', 'altes g 0.08 0.92', 'altes g 0.01 0.99', 'altes 0.01 0.98', 'CVOFLS', 'shift 1', 'alt']

# paths = ['2006020903 cds Vergleich', '2006031402 dshift1 zu viel Dämpfung', '2006031403 CVOFLS', '2006031405 shift1', '2006021958 shift1 weights 0 oder 1', '2006022001 altes curv_ml']
# labels = ['cds', 'dshift1', 'cvofls', 'shift1', 'shift1 w01', 'alt']

# paths = ['FASTEST_1', 'FASTESET_2', 'FASTEST_3', 'FASTEST_4', '2006031403 CVOFLS']
# labels = ['F1', 'F2', 'F3', 'F4', 'CVOFLS']

paths = ['FASTEST_1', 'FASTEST_2', '2006042307 dshift1 shift1 flip 0.05 sehr gut', '2006041645 dshift1 shift1 0.03 w + kappa neq 0 g sqr all ganz gut', '2006040836 altes w bei k ungleich 0 0.08 0.92 wenig Fehler kleine Amplitude', '2006031403 CVOFLS', '2006022000 cds Vergleich']
labels = ['F1', 'F2', 'flip 0.05', 'dshift1 shift1 0.03', 'altes shift1 0.08 0.92', 'CVOFLS', 'CDS']

reds = ['crimson', 'lightcoral', 'orangered', 'chocolate']
j=0
endtime = 15  # in s
timestep = 0.000625

refdata = pd.read_csv('Strubelj_128.txt', skiprows=1, names=['t', 'Referenz 1 (CVOFLS Paper)'])
refdata['t'] = refdata['t']*1/timestep
refdata = refdata.set_index('t')
refdata.plot(ax=ax, label='CVOFLS (Paper)', color='darkgray', lw=2, ls='-')

for i in range(len(paths)):
    path = paths[i]
    label = labels[i]
    data = pd.read_csv(os.path.join(path, 'y_pos_re.txt'), skiprows=1, names=['it', 'x', 'y', 'c', 'cm'])
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

    if (('CVOFLS' in label)):
        color = 'dimgray'
    elif (re.match(r'F\d', label) or ('altes' in label)):
        color = reds[j]
        j = j+1
    elif ('CDS' in label):
        # color = 'steelblue'
        color = 'cyan'
    else:
        color = None

    pd.Series(y_05).plot(label=label, color=color)


ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: f'{x*timestep}'))
plt.xticks(np.arange(0, 11/timestep, 2/timestep))
ax.set_ylabel('y-pos')
ax.set_xlabel('time [s]')

for i in range(int(np.floor(2*endtime/1.5888)+1)):
    plt.axvline(x=i/2*1.5888/timestep, color='k', lw=0.5)

fig.tight_layout()
plt.legend()
plt.savefig('result.png', dpi=150)
plt.show()
