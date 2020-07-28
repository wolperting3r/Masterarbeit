import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import tikzplotlib as tkz
import regex
import re
from scipy import signal
from pprint import pprint

def cal_ypos(path):
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

    # Cut after endtime
    data = data.iloc[:np.int32(diffpos*(endtime+0.5)*1/timestep)+diffpos, :]

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
    return (y_lower + (c_mid-c_lower)/(c_upper-c_lower)*(y_upper-y_lower))

# fig, ax = plt.subplots(1, 1, figsize=(10,5))
# Make figure without border
fig = plt.figure(frameon=False)
fig.set_size_inches(10,5)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

data = 'int1'
percentiles = True

# '''
if data == 'int0':
    # Vergleich selbes Netz int 0
    paths = [ '2007212130 int 0 s1', '2007212130 int 0 s2', '2007212130 int 0 s3', '2007212130 int 0 s4', '2007212130 int 0 s5', '2007212130 int 0 s6', '2007212130 int 0 s7', '2007212130 int 0 s8', '2007212130 int 0 s9', '2007212130 int 0 s10', '2007212130 int 0 s11', '2007212130 int 0 s12', '2007212130 int 0 s13', '2007212130 int 0 s14', '2007212130 int 0 s15', '2007212130 int 0 s16', '2006022000 cds Vergleich', '2006031403 CVOFLS']
    labels = [ '2007212130 int 0 s1', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', 'CV', 'CVOFLS' ]
elif data == 'int1':
    # Vergleich selbes Netz int 1
    paths = [ '2007221249 int 1 s1', '2007221249 int 1 s2', '2007221249 int 1 s3', '2007221249 int 1 s4', '2007221249 int 1 s5', '2007221249 int 1 s6', '2007221249 int 1 s7', '2007221249 int 1 s8', '2007221249 int 1 s9', '2007221249 int 1 s10', '2007221249 int 1 s11', '2007221249 int 1 s12', '2007221249 int 1 s13', '2007221249 int 1 s14', '2007221249 int 1 s15', '2007221249 int 1 s16', '2006022000 cds Vergleich', '2006031403 CVOFLS' ]
    labels = [ '2007221249 int 1 s1', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', 'CV', 'CVOFLS' ]
elif data == 'int2':
    # Vergleich selbes Netz int 2
    paths = [ '2007211250 int 2 s1', '2007211250 int 2 s2', '2007211250 int 2 s3', '2007211250 int 2 s4', '2007211250 int 2 s5', '2007211250 int 2 s6', '2007211250 int 2 s7', '2007211250 int 2 s8', '2007211250 int 2 s9', '2007211250 int 2 s10', '2007211250 int 2 s11', '2007211250 int 2 s12', '2007211250 int 2 s13', '2007211250 int 2 s14', '2007211250 int 2 s15', '2007211250 int 2 s16', '2006022000 cds Vergleich', '2006031403 CVOFLS' ]
    labels = [ '2007211250 int 2 s1', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_', 'CV', 'CVOFLS' ]

reds = ['tomato', 'red', 'maroon', 'brown']
j=0
k=0
endtime = 2  # in s
timestep = 0.000625

refdata = pd.read_csv('Strubelj_128.txt', skiprows=1, names=['t', 'CVOFLS Paper'])
refdata['t'] = refdata['t']*1/timestep
refdata = refdata.set_index('t')
# refdata.plot(ax=ax, label='CVOFLS (Paper)', color='darkgray', lw=2, ls='-')

colors=iter(plt.cm.rainbow(np.linspace(0,1,len(paths))))
y_05s = {}
# Get y-position of interface for all paths
for path in paths:
    if ('CVOFLS' in path):
        cvofls_pos = cal_ypos(path)
    if ('cds' in path):
        cds_pos = cal_ypos(path)
    else:
        y_05s[paths.index(path)] = cal_ypos(path)

# Make numpy array with all values
y_pos = np.empty((y_05s[0].shape[0], len(y_05s.keys())))
for key in y_05s.keys():
    y_pos[:, list(y_05s.keys()).index(key)] = y_05s[key]

# Get filter parameters
b, a = signal.butter(3, 0.005, btype='low', analog=False)

# Calculate, filter and plot median
median = np.median(y_pos, axis=1)
median = signal.filtfilt(b, a, median)

color_cvofls = 'olivedrab'
color_cds = 'cadetblue'
if data == 'int2':
    color_ml = 'chocolate'  # int 2
elif data == 'int1':
    color_ml = 'goldenrod'  # int 1
elif data == 'int0':
    color_ml = 'crimson'  # int 0

lsp = np.linspace(0, median.shape[0], median.shape[0])
n_val = 10
labels = iter(['10%-Percentile'] + ['_nolabel_']*(n_val))
if percentiles:
    for i in range(n_val):
        # Calculate, filter and plot lower to upper percentil
        up_perc = np.percentile(y_pos, 100-(i)/n_val*50, axis=1, interpolation='midpoint')
        lo_perc = np.percentile(y_pos, (i)/n_val*50, axis=1, interpolation='midpoint')

        up_perc = signal.filtfilt(b, a, up_perc)
        lo_perc = signal.filtfilt(b, a, lo_perc)

        # plt.fill_between(lsp, lo_perc, up_perc, alpha=0.8/n_val, color=color_ml, label=next(labels))
        plt.fill_between(lsp, lo_perc, up_perc, alpha=0.8/n_val, color=color_ml)

labels = iter(['ANNs'] + ['_nolabel_']*(len(y_05s.keys())))
# '''
for key in y_05s.keys():
    # Plot y-positions
    if percentiles:
        plt.plot(y_05s[key], c=color_ml, alpha = 0.3, lw=0.5, label=next(labels))  # Percentile
    else:
        plt.plot(y_05s[key], c=color_ml, alpha = 0.5, lw=1, label=next(labels))  # Nur Linien
# '''

# Plot median (50%-Percentil)
if percentiles:
    plt.plot(median, c=color_ml, lw=2, label='ANN Median')
plt.plot(cvofls_pos, c=color_cvofls, label='CVOFLS')
plt.plot(cds_pos, c=color_cds, label='CDS')


'''
ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: f'{x*timestep}'))
plt.xticks(np.arange(start=0, step=1.5888/(2*timestep), stop=int((endtime))/timestep))

ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, pos: '%.2E' % np.round((y-0.075/2)*0.1, 4)))
plt.yticks(np.arange(start=(0.002*10)+0.075/2, step=0.005 , stop=(0.003*10)+0.075/2))
ax.set_ylabel('y-Pos [m]')
ax.set_xlabel('Zeit [s]')
# '''
ax.set_ylim([0.0540,0.0661])
ax.set_xlim([0, ((endtime)/timestep)])

for i in range(int(np.floor(2*endtime/1.5888)+1)):
    plt.axvline(x=i/2*1.5888/timestep, color='k', lw=0.5)

# fig.tight_layout()
# Set opacity of entries in legend to 1
'''
leg = plt.legend()
for lh in leg.legendHandles: 
    # pprint(vars(lh))
    if len(lh.get_label()) == 0:  # Percentile
        lh.set_alpha(0.3)
    else:
        lh.set_alpha(1)
# '''
# '''
# '''
# Export tikz file
# tkz.save('result.tex', axis_height='7cm', axis_width='15cm', extra_axis_parameters={'scaled y ticks=manual:{$\cdot10^{-3}$}{\pgfmathparse{#1-1}}'})
# '''
tkz.save('result.tex', axis_height='7cm', axis_width='15cm')
with open('result.tex', 'r') as myfile:
    data = myfile.read()
    data = re.sub('semithick', 'ultra thick', data)
    data = regex.sub(r'(?<=yticklabels\=\{.*)(\d\.\d)\d*E\-03', r'\1', data)

with open('result.tex', 'w') as myfile:
    myfile.write(data)

plt.savefig('result.svg', dpi=150)
os.system('inkscape -D result.svg -o result.pdf --export-latex')
# '''
# plt.show()
