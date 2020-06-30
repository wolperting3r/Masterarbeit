from src.execution import exe_ml, exe_save
from src.execution import exe_dg

''' Machine Learning '''

''' Train '''
# Network related
epochs = [1000]
stencil = [[7, 7]]
activation = ['relu']
learning_rate = [1e-4]
hf = ['hf']
hf_correction = [False]
dropout = [0]

angle = [False]
rot = [True]
flip = [True]
cut = [True]
shift = [1]
bias = [True]
edge = [0]
custom_loss = [False]
addstring = ['_1', '_2', '_3', '_4']
# addstring = ['']
# addstring = ['_1', '_2']
# addstring = ['_4']

# Data related
neg = [True]
# model_mlp_1000_200-150-120_7x7_rot_flp_cut_dshift1_shift1_bia_int2
# load_data = ['data_CVOFLS_7x7_g2_eqk2']
# load_data = ['data_CVOFLS_7x7']
load_data = ['']
# interpolate = [0, 1, 1.5, 2]
interpolate = [1]
data = ['ellipse']
plotdata = ['ellipse']
smearing = [True]
dshift = ['1']  # 0, 1, 1b
gauss = [1]

# '''
# 1.: Train, 2.: Plot
for i in range(0, 2):
    # MLP
    network = ['mlp']
    # layer = [[200, 150, 120]]
    # layer = [[500, 300, 150]]
    layer = [[200, 500, 1000, 500, 150]]
    # layer = [[100, 80], [250, 150]]
    # layer = [[200, 180, 150, 120, 100]]
    # layer = [[100, 80], [200, 150, 120]]

    if i == 0:
        plot = [False]
        exe_ml(plot=plot, network=network, stencil=stencil, layer=layer, activation=activation, epochs=epochs, learning_rate=learning_rate, neg=neg, angle=angle, rot=rot, data=data, smearing=smearing, hf=hf, hf_correction=hf_correction, dropout=dropout, plotdata=plotdata, addstring=addstring, flip=flip, cut=cut, dshift=dshift, shift=shift, bias=bias, edge=edge, custom_loss=custom_loss, gauss=gauss, load_data=load_data, interpolate=interpolate)
    if i == 1:
        plot = [True]
        exe_ml(plot=plot, network=network, stencil=stencil, layer=layer, activation=activation, epochs=epochs, learning_rate=learning_rate, neg=neg, angle=angle, rot=rot, data=data, smearing=smearing, hf=hf, hf_correction=hf_correction, dropout=dropout, plotdata=plotdata, addstring=addstring, flip=flip, cut=cut, dshift=dshift, shift=shift, bias=bias, edge=edge, custom_loss=custom_loss, gauss=gauss, load_data=load_data, interpolate=interpolate)
        exe_save(plot=plot, network=network, stencil=stencil, layer=layer, activation=activation, epochs=epochs, learning_rate=learning_rate, neg=neg, angle=angle, rot=rot, data=data, smearing=smearing, hf=hf, hf_correction=hf_correction, dropout=dropout, plotdata=plotdata, addstring=addstring, flip=flip, cut=cut, dshift=dshift, shift=shift, bias=bias, edge=edge, custom_loss=custom_loss, gauss=gauss, load_data=load_data, interpolate=interpolate)
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
stencils = [[7, 7]]
# stencils = [[5, 5]]
# stencils = [[9, 9]]
ek = [True]
neg = [True]
# N_values = [7e6]
N_values = [1e6]
# N_values = [1]
silent = [False]
# geometry = ['sinus', 'ellipse']
geometry = ['ellipse']
smearing = [True]
usenormal = [True]
gauss = [True]
# interpolate = [0, 1, 1.5, 2]
interpolate = [0, 1, 1.5, 2]
# interpolate = [0]
# interpolate = [2]
exe_dg(stencils=stencils, ek=ek, neg=neg, N_values=N_values, silent=silent, geometry=geometry, smearing=smearing, usenormal=usenormal, interpolate=interpolate, gauss=gauss)
# '''
