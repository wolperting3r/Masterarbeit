from src.execution import exe_ml, exe_save
from src.execution import exe_dg

''' Machine Learning '''

''' Train '''
epochs = [1000]
stencil = [[9, 9]]
activation = ['relu']
learning_rate = [1e-4]
neg = [True]
angle = [False]
rot = [True]
data = ['ellipse']
plotdata = ['ellipse']
smearing = [True]
hf = ['hf']
hf_correction = [False]
dropout = [0]
flip = [True]
cut = [True]
dshift = ['1']  # 0, 1, 1b
shift = [1]
bias = [True]
# interpolate = [1.5, 2]
# interpolate = [0, 1, 1.5, 2]
interpolate = [0]
edge = [1]
custom_loss = [False]
# addstring = ['_1', '_2', '_3', '_4']
# addstring = ['_5']
# addstring = ['_6']
# addstring = ['_7']
# addstring = ['_8']
# addstring = ['_n_1', '_n_2']
addstring = ['']

# '''
# 1.: Train, 2.: Plot
for i in range(0, 2):
    # MLP
    network = ['mlp']
    layer = [[200, 150, 120]]
    # layer = [[100, 80], [200, 150, 120]]

    if i == 0:
        plot = [False]
        # exe_ml(plot=plot, network=network, stencil=stencil, layer=layer, activation=activation, epochs=epochs, learning_rate=learning_rate, neg=neg, angle=angle, rot=rot, data=data, smearing=smearing, hf=hf, hf_correction=hf_correction, dropout=dropout, plotdata=plotdata, addstring=addstring, flip=flip, cut=cut, dshift=dshift, shift=shift, bias=bias, edge=edge, custom_loss=custom_loss, interpolate=interpolate)
    if i == 1:
        plot = [True]
        exe_ml(plot=plot, network=network, stencil=stencil, layer=layer, activation=activation, epochs=epochs, learning_rate=learning_rate, neg=neg, angle=angle, rot=rot, data=data, smearing=smearing, hf=hf, hf_correction=hf_correction, dropout=dropout, plotdata=plotdata, addstring=addstring, flip=flip, cut=cut, dshift=dshift, shift=shift, bias=bias, edge=edge, custom_loss=custom_loss, interpolate=interpolate)
        # exe_save(plot=plot, network=network, stencil=stencil, layer=layer, activation=activation, epochs=epochs, learning_rate=learning_rate, neg=neg, angle=angle, rot=rot, data=data, smearing=smearing, hf=hf, hf_correction=hf_correction, dropout=dropout, plotdata=plotdata, addstring=addstring, flip=flip, cut=cut, dshift=dshift, shift=shift, bias=bias, edge=edge, custom_loss=custom_loss, interpolate=interpolate)
    # '''

'''
# CVN
network = ['cvn']
layer = [[32], [32, 64]]
if i == 0:
    plot = [False]
    exe_ml(plot=plot, network=network, stencil=stencil, layer=layer, activation=activation, epochs=epochs, learning_rate=learning_rate, neg=neg, angle=angle, rot=rot, data=data, smearing=smearing, hf=hf, hf_correction=hf_correction, dropout=dropout)
# '''
'''
if i == 1:
    plot = [True]
    exe_ml(plot=plot, network=network, stencil=stencil, layer=layer, activation=activation, epochs=epochs, learning_rate=learning_rate, neg=neg, angle=angle, rot=rot, data=data, smearing=smearing, hf=hf, hf_correction=hf_correction, dropout=dropout, plotdata=plotdata)
# '''
'''
# Autoencoder
# 5x5
network = ['auto']
stencils = [[5, 5]]
layers = [[6, 3, 20], [20, 10, 80]]
activation = ['relu']
if i == 0:
    exe_ml(network=network, stencils=stencils, layers=layers, activation=activation, epochs=epochs, learning_rate=learning_rate, neg=neg, angle=angle, rot=rot, data=data, smearing=smearing, hf=hf, hf_correction=hf_correction, dropout=dropout)
elif i == 1:
    exe_ml_plot(network=network, stencils=stencils, layers=layers, activation=activation, epochs=epochs, learning_rate=learning_rate, neg=neg, angle=angle, rot=rot, data=data, smearing=smearing, hf=hf, hf_correction=hf_correction, dropout=dropout)
# '''

''' Data Generation '''
'''
# stencils = [[7, 7]]
# stencils = [[5, 5]]
stencils = [[9, 9]]
ek = [True]
neg = [True]
# N_values = [7e6]
N_values = [1e6]
# N_values = [1e2]
silent = [False]
# geometry = ['sinus', 'ellipse']
geometry = ['ellipse']
smearing = [True]
usenormal = [True]
# interpolate = [0, 1, 1.5, 2]
interpolate = [0, 1, 1.5, 2]
# interpolate = [0]
# interpolate = [2]
exe_dg(stencils=stencils, ek=ek, neg=neg, N_values=N_values, silent=silent, geometry=geometry, smearing=smearing, usenormal=usenormal, interpolate=interpolate)
# '''
