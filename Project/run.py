from src.execution import exe_ml, exe_save
from src.execution import exe_dg

''' Machine Learning '''

''' Train '''
# Network related
epochs = [500]
batch_size = [128]
# stencil = [[9, 9]]
# stencil = [[3, 3], [5, 5], [7, 7], [9, 9]] 
stencil = [[7, 7]] 
activation = ['relu']
learning_rate = [1e-5]
hf = ['hf']
hf_correction = [False]
dropout = [0]

angle = [False]
rot = [True]
flip = [False]
cut = [True]
shift = [1]
bias = [True]
edge = [0]
edge2 = [True]
unsharp_mask = [False]
amount = [0.01]
custom_loss = [False]
seed = [1]
# seed = [1, 2, 3, 4]
addstring = ['_fkernel']

# Data related
neg = [True]
# model_mlp_1000_200-150-120_7x7_rot_flp_cut_dshift1_shift1_bia_int2
# load_data = ['data_CVOFLS_7x7_f_eqk']
load_data = ['']
interpolate = [2]
data = ['ellipse']
plotdata = ['ellipse']
smearing = [True]
dshift = [1]  # 0, 1, '1b'
gauss = [0]
network = ['mlp']
layer = [[200, 150, 120]]

if __name__ == '__main__':
   #  '''
    # 1.: Train, 2.: Plot
    for i in range(0, 2):
        if i == 0:
            plot = [False]
            exe_ml(plot=plot, network=network, stencil=stencil, layer=layer, batch_size=batch_size, activation=activation, epochs=epochs, learning_rate=learning_rate, neg=neg, angle=angle, rot=rot, data=data, smearing=smearing, hf=hf, hf_correction=hf_correction, dropout=dropout, plotdata=plotdata, addstring=addstring, flip=flip, cut=cut, dshift=dshift, shift=shift, bias=bias, edge=edge, edge2=edge2, custom_loss=custom_loss, gauss=gauss, load_data=load_data, seed=seed, interpolate=interpolate, unsharp_mask=unsharp_mask, amount=amount)
        if i == 1:
            plot = [True]
            # exe_ml(plot=plot, network=network, stencil=stencil, layer=layer, batch_size=batch_size, activation=activation, epochs=epochs, learning_rate=learning_rate, neg=neg, angle=angle, rot=rot, data=data, smearing=smearing, hf=hf, hf_correction=hf_correction, dropout=dropout, plotdata=plotdata, addstring=addstring, flip=flip, cut=cut, dshift=dshift, shift=shift, bias=bias, edge=edge, edge2=edge2, custom_loss=custom_loss, gauss=gauss, load_data=load_data, seed=seed, interpolate=interpolate, unsharp_mask=unsharp_mask, amount=amount)
            # exe_save(plot=plot, network=network, stencil=stencil, layer=layer, batch_size=batch_size, activation=activation, epochs=epochs, learning_rate=learning_rate, neg=neg, angle=angle, rot=rot, data=data, smearing=smearing, hf=hf, hf_correction=hf_correction, dropout=dropout, plotdata=plotdata, addstring=addstring, flip=flip, cut=cut, dshift=dshift, shift=shift, bias=bias, edge=edge, edge2=edge2, custom_loss=custom_loss, gauss=gauss, load_data=load_data, seed=seed, interpolate=interpolate, unsharp_mask=unsharp_mask, amount=amount)
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
    stencils = [[3, 3], [5, 5], [7, 7], [9, 9]]
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
