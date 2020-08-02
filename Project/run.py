from src.execution import exe_ml, exe_save
from src.execution import exe_dg

''' Machine Learning '''

''' Train '''
# Network related
epochs = [1000]
batch_size = [128]
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
edge = [1]
custom_loss = [False]
seed = [1]
# seed = [1, 2]
# seed = [3, 4]
# seed = [1, 2, 3, 4]
# seed = [5, 6, 7, 8]
# seed = [9, 10, 11, 12]
# seed = [13, 14, 15, 16]
# seed = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# seed = [7, 8]
# addstring = ['_1', '_2', '_3', '_4']
# addstring = ['_cut0.37']
# addstring = ['_1', '_2']
# addstring = ['_4']
# addstring = ['_cut0109']
addstring = ['_neuEdge']

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
dshift = [1]  # 0, 1, '1b'
gauss = [1]

if __name__ == '__main__':
    # '''
    # 1.: Train, 2.: Plot
    for i in range(0, 2):
        # MLP
        network = ['mlp']
        # layer = [[200, 150, 120]]
        layer = [[200, 200, 200, 200, 200]]
        # layer = [[500, 300, 150]]
        # layer = [[200, 500, 1000, 500, 150]]
        # layer = [[100, 80], [250, 150]]
        # layer = [[200, 180, 150, 120, 100]]
        # layer = [[100, 80], [200, 150, 120]]
        # layer = [[80], [120], [100, 80], [200, 150], [200, 150, 120]]
        # layer = [[100, 80, 50], [100, 100, 100], [150, 120, 100]]
        # layer = [[80], [120], [100, 80], [200, 150], [200, 150, 120], [100, 80, 50], [100, 100, 100], [150, 120, 100]]

        if i == 0:
            plot = [False]
            exe_ml(plot=plot, network=network, stencil=stencil, layer=layer, batch_size=batch_size, activation=activation, epochs=epochs, learning_rate=learning_rate, neg=neg, angle=angle, rot=rot, data=data, smearing=smearing, hf=hf, hf_correction=hf_correction, dropout=dropout, plotdata=plotdata, addstring=addstring, flip=flip, cut=cut, dshift=dshift, shift=shift, bias=bias, edge=edge, custom_loss=custom_loss, gauss=gauss, load_data=load_data, seed=seed, interpolate=interpolate)
        if i == 1:
            plot = [True]
            exe_ml(plot=plot, network=network, stencil=stencil, layer=layer, batch_size=batch_size, activation=activation, epochs=epochs, learning_rate=learning_rate, neg=neg, angle=angle, rot=rot, data=data, smearing=smearing, hf=hf, hf_correction=hf_correction, dropout=dropout, plotdata=plotdata, addstring=addstring, flip=flip, cut=cut, dshift=dshift, shift=shift, bias=bias, edge=edge, custom_loss=custom_loss, gauss=gauss, load_data=load_data, seed=seed, interpolate=interpolate)
            exe_save(plot=plot, network=network, stencil=stencil, layer=layer, batch_size=batch_size, activation=activation, epochs=epochs, learning_rate=learning_rate, neg=neg, angle=angle, rot=rot, data=data, smearing=smearing, hf=hf, hf_correction=hf_correction, dropout=dropout, plotdata=plotdata, addstring=addstring, flip=flip, cut=cut, dshift=dshift, shift=shift, bias=bias, edge=edge, custom_loss=custom_loss, gauss=gauss, load_data=load_data, seed=seed, interpolate=interpolate)
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
    geometry = ['ellipse', 'circle']
    smearing = [False]
    usenormal = [True]
    dshift = [False]
    gauss = [False]
    # interpolate = [1, 1.5, 2]
    interpolate = [0]
    # interpolate = [0]
    # interpolate = [2]
    exe_dg(stencils=stencils, ek=ek, neg=neg, N_values=N_values, silent=silent, geometry=geometry, smearing=smearing, usenormal=usenormal, dshift=dshift, interpolate=interpolate, gauss=gauss)
    # '''
