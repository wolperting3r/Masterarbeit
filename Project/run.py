from src.execution import exe_ml, exe_save
#from src.execution import exe_dg

''' Data Generation '''
'''
stencils = [[7, 7]]
# stencils = [[5, 5]]
# stencils = [[9, 9]]
ek = [True]
neg = [True]
N_values = [1e6]
# N_values = [1e3]
# N_values = [2]
silent = [False]
# geometry = ['sinus', 'ellipse']
geometry = ['circle']
smearing = [True]
usenormal = [True, False]
exe_dg(stencils=stencils, ek=ek, neg=neg, N_values=N_values, silent=silent, geometry=geometry, smearing=smearing, usenormal=usenormal)
# '''

''' Machine Learning '''

''' Train '''
epochs = [1000]
# epochs = [1]
# stencils = [[7, 7], [3, 3], [5, 5]]
stencil = [[7, 7]]
# stencil = [[5, 5]]
# stencil = [[9, 9]]
activation = ['relu']
learning_rate = [1e-4]
neg = [True]
angle = [False]
rot = [True]
# data = ['sinus']
# data = ['all', 'sinus', 'circle', 'ellipse']
# plotdata = ['all', 'sinus', 'circle', 'ellipse']
# data = ['all']
data = ['ellipse']
plotdata = ['all']
smearing = [True]
hf = ['hf']
hf_correction = [False]
dropout = [0]
addstring = ['_nocust_flip_dshift1_shift1_cut0050995']
# 1.: Train, 2.: Plot
for i in range(0, 2):
    # CVN
    network = ['cvn']
    layer = [[32], [32, 64]]
    '''
    if i == 0:
        plot = [False]
        exe_ml(plot=plot, network=network, stencil=stencil, layer=layer, activation=activation, epochs=epochs, learning_rate=learning_rate, neg=neg, angle=angle, rot=rot, data=data, smearing=smearing, hf=hf, hf_correction=hf_correction, dropout=dropout)
    # '''
    '''
    if i == 1:
        plot = [True]
        exe_ml(plot=plot, network=network, stencil=stencil, layer=layer, activation=activation, epochs=epochs, learning_rate=learning_rate, neg=neg, angle=angle, rot=rot, data=data, smearing=smearing, hf=hf, hf_correction=hf_correction, dropout=dropout, plotdata=plotdata)
    # '''

    # MLP
    network = ['mlp']
    # layer = [[100, 80], [80], [50, 50], [50, 40, 30]]
    # layer = [[100, 80], [100, 80, 50]]
    # layer = [[200, 150, 120]]
    # layer = [[200, 150, 120, 80, 50]]
    # layer = [[500, 600, 400, 100]]
    layer = [[100, 80]]
    # layer = [[80, 50]]
    # '''
    if i == 0:
        plot = [False]
        exe_ml(plot=plot, network=network, stencil=stencil, layer=layer, activation=activation, epochs=epochs, learning_rate=learning_rate, neg=neg, angle=angle, rot=rot, data=data, smearing=smearing, hf=hf, hf_correction=hf_correction, dropout=dropout, plotdata=plotdata, addstring=addstring)
    # '''
    # '''
    if i == 1:
        plot = [True]
        exe_ml(plot=plot, network=network, stencil=stencil, layer=layer, activation=activation, epochs=epochs, learning_rate=learning_rate, neg=neg, angle=angle, rot=rot, data=data, smearing=smearing, hf=hf, hf_correction=hf_correction, dropout=dropout, plotdata=plotdata, addstring=addstring)
        exe_save(plot=plot, network=network, stencil=stencil, layer=layer, activation=activation, epochs=epochs, learning_rate=learning_rate, neg=neg, angle=angle, rot=rot, data=data, smearing=smearing, hf=hf, hf_correction=hf_correction, dropout=dropout, plotdata=plotdata, addstring=addstring)
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
