import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import tikzplotlib as tkz
import regex
import re

fig, ax = plt.subplots(1, 1, figsize=(10,5))

# '''
# Fastest Auswertung
paths = [
    # '2008031931 Edge neu Relaxation FNB Ellipse s1',
    '2008031931 Edge neu Relaxation FNB Ellipse s2',
    '2008031931 Edge neu Relaxation FNB Ellipse s3',
    '2008031931 Edge neu Relaxation FNB Ellipse s4',
    '2008031931 Edge neu Relaxation FNB Ellipse s5',
    '2008031931 Edge neu Relaxation FNB Ellipse s6',
    '2008031931 Edge neu Relaxation FNB Ellipse s7',
    '2008031931 Edge neu Relaxation FNB Ellipse s8',
    '2008031931 Edge neu Relaxation FNB Ellipse s9',
    '2008031931 Edge neu Relaxation FNB Ellipse s10',
    '2008031931 Edge neu Relaxation FNB Ellipse s11',
    '2008031931 Edge neu Relaxation FNB Ellipse s12',
    '2008031931 Edge neu Relaxation FNB Ellipse s13',
    '2008031931 Edge neu Relaxation FNB Ellipse s14',
    'FASTEST_1',
    # 'FASTEST_2',
    # 'FASTEST_3',
    # 'FASTEST_4',
    '2006031403 CVOFLS',
    '2006022000 cds Vergleich',
]
labels = [
    # '2008031931 Edge neu Relaxation FNB Ellipse s1',
    '2008031931 Edge neu Relaxation FNB Ellipse s2',
    '2008031931 Edge neu Relaxation FNB Ellipse s3',
    '2008031931 Edge neu Relaxation FNB Ellipse s4',
    '2008031931 Edge neu Relaxation FNB Ellipse s5',
    '2008031931 Edge neu Relaxation FNB Ellipse s6',
    '2008031931 Edge neu Relaxation FNB Ellipse s7',
    '2008031931 Edge neu Relaxation FNB Ellipse s8',
    '2008031931 Edge neu Relaxation FNB Ellipse s9',
    '2008031931 Edge neu Relaxation FNB Ellipse s10',
    '2008031931 Edge neu Relaxation FNB Ellipse s11',
    '2008031931 Edge neu Relaxation FNB Ellipse s12',
    '2008031931 Edge neu Relaxation FNB Ellipse s13',
    '2008031931 Edge neu Relaxation FNB Ellipse s14',
    'F1',
    # 'F2',
    # 'F3',
    # 'F4',
    'CVOFLS',
    'CDS Vergleich',
]
#'''

'''
# CVOFLS ML
paths = [
    '2006031403 CVOFLS',
    '2006300749 cvofls ml viele Daten 1 am ungenauesten trainiert schlechtestes Ergebnis/',
    '2006300749 cvofls ml viele Daten 2 am genauesten trainiert bestes Ergebnis/',
    '2006300749 cvofls ml viele Daten 3/',
    '2006300749 cvofls ml viele Daten 4/',
    # '2006221540 cvofls ml g2 7x7 1/',
    # '2006221540 cvofls ml g2 7x7 2/',
    # '2006230852 cvofls ml g2 7x7 2x g 1/',
    # '2006230852 cvofls ml g2 7x7 2x g 2/',
    '2006230852 cvofls ml g2 9x9 1x g 1/',
    '2006230852 cvofls ml g2 9x9 1x g 2/',
    '2006231130 cvofls ml g2 7x7 2x g 1 g01g005/',
    '2006231130 cvofls ml g2 7x7 2x g 2 g01g005/',
    '2006231130 cvofls ml g2 7x7 wg 001 neue Daten 1/',
    '2006231130 cvofls ml g2 7x7 wg 001 neue Daten 2/',
    '2006231423 cvofls ml g2 7x7 wg 001 neue Daten 3/',
    '2006231423 cvofls ml g2 7x7 wg 001 neue Daten 4/',
    # '2006022000 cds Vergleich',
]
labels = [
    'CVOFLS',
    'CVOFLS ML viele Daten 1',
    'CVOFLS ML viele Daten 2 (MSE am kleinsten)',
    'CVOFLS ML viele Daten 3 (MSE am größten)',
    'CVOFLS ML viele Daten 4',
    # '2006221540 cvofls ml g2 7x7 1/',
    # '2006221540 cvofls ml g2 7x7 2/',
    # '2006230852 cvofls ml g2 7x7 2x g 1/',
    # '2006230852 cvofls ml g2 7x7 2x g 2/',
    'CVOFLS ML Datensatz 1',
    'CVOFLS ML Datensatz 1x',
    'CVOFLS ML Datensatz 1x',
    'CVOFLS ML Datensatz 1x',
    'CVOFLS ML Datensatz 2',
    'CVOFLS ML Datensatz 2x',
    'CVOFLS ML Datensatz 2x',
    'CVOFLS ML Datensatz 2x',
    # 'CDS',
]
# '''

'''
# Vergleich Scharfes Interface
paths = [
    '2006031403 CVOFLS',
    '2006022000 cds Vergleich',
    '2006182006 edge int0 gauss/',
    '2006182006 edge int2 gauss/',
]
labels = [
    'CVOFLS',
    'CDS',
    'Scharfes Interface 1',
    'Scharfes Interface 2',
]
# '''

'''
# Vergleich selbes Netz int 0
paths = [
    '2007011530 int 0 1',
    '2007011530 int 0 2',
    '2007011530 int 0 3',
    '2007011530 int 0 4',
    '2007011530 int 0 5',
    '2007011530 int 0 6',
    '2006031403 CVOFLS'
]
labels = [
    'ANNs',
    '_nolegend_',
    '_nolegend_',
    '_nolegend_',
    '_nolegend_',
    '_nolegend_',
    'CVOFLS'
]
#'''

'''
# Vergleich selbes Netz int 2
paths = [
    '2007011530 int 2 1',
    '2007011530 int 2 2',
    '2007011530 int 2 3',
    '2007011530 int 2 4',
    '2007011530 int 2 5',
    '2007011530 int 2 6',
    '2006031403 CVOFLS'
]
labels = [
    'ANNs',
    '_nolegend_',
    '_nolegend_',
    '_nolegend_',
    '_nolegend_',
    '_nolegend_',
    'CVOFLS'
]
#'''

# reds = ['maroon', 'brown', 'indianred', 'lightcoral']
# reds = ['maroon', 'red', 'tomato', 'lightsalmon', 'maroon', 'red', 'tomato', 'lightsalmon', 'maroon', 'red', 'tomato', 'lightsalmon', 'maroon', 'red', 'tomato', 'lightsalmon',]
'''
reds = ['red', 'peru', 'darkviolet', 'green',
        'red', 'peru', 'darkviolet', 'green',
        'red', 'peru', 'darkviolet', 'green',
        'red', 'peru', 'darkviolet', 'green',
        'red', 'peru', 'darkviolet', 'green',
        'red', 'peru', 'darkviolet', 'green',
       ]
# '''
reds = ['tomato', 'red', 'maroon', 'brown']
j=0
k=0
endtime = 3  # in s
timestep = 0.000625

refdata = pd.read_csv('Strubelj_128.txt', skiprows=1, names=['t', 'CVOFLS Paper'])
refdata['t'] = refdata['t']*1/timestep
refdata = refdata.set_index('t')
# refdata.plot(ax=ax, label='CVOFLS (Paper)', color='darkgray', lw=2, ls='-')

colors=iter(plt.cm.rainbow(np.linspace(0,1,len(paths))))
for i in range(len(paths)):
    path = paths[i]
    label = labels[i]
    filepath = os.path.join(path, 'y_pos_re.txt')
    filepath_ext = os.path.join('/Volumes','Daten','Archive','Masterarbeit','fortran_test','dynamic','model_mlp_1000_100-80_7x7_eqk_0.0001_128_relu_neg_nag_rot_all_smr_nhc_128_allstc', path, 'y_pos_re.txt')
    if os.path.isfile(filepath):
        data = pd.read_csv(filepath, skiprows=1, names=['it', 'x', 'y', 'c', 'cm'])
    elif os.path.isfile(filepath_ext):
        data = pd.read_csv(filepath_ext, skiprows=1, names=['it', 'x', 'y', 'c', 'cm'])
    else:
        print(f'file {path} not found!')

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

    if (label == 'CVOFLS') or ('MSE' in label):
        alpha = 1
    else:
        alpha = 0.5
    linewidth = 1.5

    if (re.match(r'F\d', label) or ('altes' in label)):
        color = reds[j]
        # color = 'gold'
        j = j+1
        linewidth = 2
    elif re.match(r'2007011530 int 0', path):
        color = 'red'
        # label = '_nolegend_'
    elif re.match(r'2007011530 int 1(?!=\.)', path):
        color = 'darkviolet'
    elif re.match(r'2007011530 int 1.5', path):
        color = 'darkviolet'
    elif re.match(r'2007011530 int 2(?!=\ss)', path):
        color = 'darkviolet'
        # color = next(colors)
    elif re.match(r'2008031037', path):
        color = 'darkviolet'
    elif re.match(r'2008031931', path):
        color = 'fuchsia'
    elif re.match(r'2007211250 int 2', path):
        color = 'cyan'
        # color = next(colors)
    elif re.match(r'2007212130 int 0', path):
        color = 'orange'
    elif re.match(r'2007221249 int 1', path):
        color = 'cyan'
    elif ('fnb' in label):
        color = 'cyan'
    elif ('2007011530' in label) or ('Netz' in label):
        color = reds[k]
        k = k+1
    elif ('cds' in label) or ('CDS' in label):
        # color = 'steelblue'
        color = 'chocolate'
    elif ('viele' in label):
        color = reds[k]
        k = k+1
    elif ('CVOFLS ML Datensatz 1' in label):
        color = 'cyan'
    elif ('CVOFLS ML Datensatz 2' in label):
        color = 'darkviolet'
    # elif re.match(r'2007011530 int 0', path):
    elif (('CVOFLS' in label)):
        # color = 'dimgray'
        color = 'lime'
    else:
        color = next(colors)

    # 'red', 'peru', 'darkviolet', 'green',
    # alpha = 0.7

    if (re.match(r'CVOFLS ML Datensatz \dx', label)):
        label = '_nolegend_'

    pd.Series(y_05).plot(label=label, c=color, alpha=alpha, linewidth=linewidth)


ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: f'{x*timestep}'))
plt.xticks(np.arange(start=0, step=1.5888/(2*timestep), stop=int((endtime))/timestep))

ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos: '%.2E' % np.round((y-0.075/2)*0.1, 4)))
plt.yticks(np.arange(start=(0.002*10)+0.075/2, step=0.005 , stop=(0.003*10)+0.075/2))
ax.set_ylabel('y-Pos [m]')
ax.set_xlabel('Zeit [s]')
# ax.set_ylim([0.0540,0.0661])

for i in range(int(np.floor(2*endtime/1.5888)+1)):
    plt.axvline(x=i/2*1.5888/timestep, color='k', lw=0.5)

fig.tight_layout()
plt.legend()
plt.savefig('result.png', dpi=150)
'''
# Export tikz file
tkz.save('result.tex', axis_height='7cm', axis_width='15cm', extra_axis_parameters={'scaled y ticks=manual:{$\cdot10^{-3}$}{\pgfmathparse{#1-1}}'})
with open('result.tex', 'r') as myfile:
    data = myfile.read()
    data = re.sub('semithick', 'ultra thick', data)
    data = regex.sub(r'(?<=yticklabels\=\{.*)(\d\.\d)\d*E\-03', r'\1', data)

with open('result.tex', 'w') as myfile:
    myfile.write(data)
# '''

plt.show()
