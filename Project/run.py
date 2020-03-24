from src.execution import exe_ml
# from src.execution import exe_dg

''' Data Generation '''
'''
stencils = [[13, 13]]
# stencils = [[5, 5], [3, 3], [7, 7]]
ek = [True]
neg = [True]
N_values = [1e6]
# N_values = [1]
silent = [False]
ellipse = [True]
smearing = [True, False]
exe_dg(stencils=stencils, ek=ek, neg=neg, N_values=N_values, silent=silent, ellipse=ellipse, smearing=smearing)
# '''


''' Machine Learning '''

''' Train '''
epochs = [250]
# stencils = [[7, 7], [3, 3], [5, 5]]
stencil = [[7, 7]]
activation = ['relu']
learning_rate = [1e-4]
neg = [True]
angle = [False]
rot = [True]
data = ['ellipse']
smearing = [True]
hf = [True]
hf_correction = [True, False]
# 1.: Train, 2.: Plot
for i in range(0, 2):
    # CVN
    network = ['cvn']
    layer = [[32], [32, 64]]
    # '''
    if i == 0:
        plot = [False]
        exe_ml(plot=plot, network=network, stencil=stencil, layer=layer, activation=activation, epochs=epochs, learning_rate=learning_rate, neg=neg, angle=angle, rot=rot, data=data, smearing=smearing, hf=hf, hf_correction=hf_correction,)
    # '''
    # '''
    if i == 1:
        plot = [True]
        exe_ml(plot=plot, network=network, stencil=stencil, layer=layer, activation=activation, epochs=epochs, learning_rate=learning_rate, neg=neg, angle=angle, rot=rot, data=data, smearing=smearing, hf=hf, hf_correction=hf_correction,)
    # '''

    # MLP
    network = ['mlp']
    # layer = [[100, 80], [80], [50, 50], [50, 40, 30]]
    layer = [[100, 80], [100, 80, 50]]
    # layer = [[100, 80], [80]]
    # '''
    if i == 0:
        plot = [False]
        exe_ml(plot=plot, network=network, stencil=stencil, layer=layer, activation=activation, epochs=epochs, learning_rate=learning_rate, neg=neg, angle=angle, rot=rot, data=data, smearing=smearing, hf=hf, hf_correction=hf_correction,)
    # '''
    # '''
    if i == 1:
        plot = [True]
        exe_ml(plot=plot, network=network, stencil=stencil, layer=layer, activation=activation, epochs=epochs, learning_rate=learning_rate, neg=neg, angle=angle, rot=rot, data=data, smearing=smearing, hf=hf, hf_correction=hf_correction,)
    # '''

    '''
    # Autoencoder
    # 5x5
    network = ['auto']
    stencils = [[5, 5]]
    layers = [[6, 3, 20], [20, 10, 80]]
    activation = ['relu']
    if i == 0:
        exe_ml(network=network, stencils=stencils, layers=layers, activation=activation, epochs=epochs, learning_rate=learning_rate, neg=neg, angle=angle, rot=rot, data=data, smearing=smearing, hf=hf, hf_correction=hf_correction,)
    elif i == 1:
        exe_ml_plot(network=network, stencils=stencils, layers=layers, activation=activation, epochs=epochs, learning_rate=learning_rate, neg=neg, angle=angle, rot=rot, data=data, smearing=smearing, hf=hf, hf_correction=hf_correction,)
    # '''
