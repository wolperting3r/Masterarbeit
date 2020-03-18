from src.execution import exe_ml, exe_ml_plot
# from src.execution import exe_dg

''' Data Generation '''
'''
# stencils = [[3, 3], [5, 5], [7, 3], [3, 7], [7, 7]]
# stencils = [[7, 3]]
stencils = [[5, 5], [3, 3], [7, 7]]
ek = [True]
neg = [True]
N_values = [1e6]
# N_values = [1e2]
silent = [False]
ellipse = [True]
smearing = [True]
exe_dg(stencils=stencils, ek=ek, neg=neg, N_values=N_values, silent=silent, ellipse=ellipse, smearing=smearing)
# '''


''' Machine Learning '''

'''
# MLP

# relu
network = ['mlp']
stencils = [[3, 3], [5, 5], [7, 3], [3, 7]]
layers = [[100], [100, 80], [100, 80, 50]]
activation = ['relu']
exe_ml(network=network, stencils=stencils, layers=layers, activation=activation)
# tanh
network = ['mlp']
stencils = [[3, 3], [5, 5], [7, 3], [3, 7]]
layers = [[100]]
activation = ['tanh']
exe_ml(network=network, stencils=stencils, layers=layers, activation=activation)
# '''

''' Train '''
epochs = [25]
# stencils = [[3, 3]]
stencils = [[7, 7]]
activation = ['relu']
learning_rate = [1e-4]
neg = [True]
angle = [False]
rot = [True]
data = ['ellipse']
smearing = [True]
# 1.: Train, 2.: Plot
for i in range(0, 2):
    # CVN
    network = ['cvn']
    layers = [[32]]
    '''
    if i == 0:
        exe_ml(
            network=network,
            stencils=stencils,
            layers=layers,
            activation=activation,
            epochs=epochs,
            learning_rate=learning_rate,
            neg=neg,
            angle=angle,
            rot=rot,
            data=data,
            smearing=smearing,
        )
    # '''
    '''
    if i == 1:
        exe_ml_plot(
            network=network,
            stencils=stencils,
            layers=layers,
            activation=activation,
            epochs=epochs,
            learning_rate=learning_rate,
            neg=neg,
            angle=angle,
            rot=rot,
            data=data,
            smearing=smearing,
        )
    # '''

    # MLP
    network = ['mlp']
    # layers = [[100, 80], [80], [50, 50], [50, 40, 30]]
    layers = [[100, 80]]
    # layers = [[80]]
    # '''
    if i == 0:
        exe_ml(
            network=network,
            stencils=stencils,
            layers=layers,
            activation=activation,
            epochs=epochs,
            learning_rate=learning_rate,
            neg=neg,
            angle=angle,
            rot=rot,
            data=data,
            smearing=smearing,
        )
    # '''
    '''
    if i == 1:
        exe_ml_plot(
            network=network,
            stencils=stencils,
            layers=layers,
            activation=activation,
            epochs=epochs,
            learning_rate=learning_rate,
            neg=neg,
            angle=angle,
            rot=rot,
            data=data,
            smearing=smearing,
        )
    # '''

    '''
    # Autoencoder
    # 3x3
    network = ['auto']
    stencils = [[3, 3]]
    layers = [[6, 3, 5], [6, 3, 80]]
    activation = ['relu', 'tanh']
    if i == 0:
        exe_ml(
            network=network,
            stencils=stencils,
            layers=layers,
            activation=activation,
            epochs=epochs,
            learning_rate=learning_rate,
            neg=neg,
            angle=angle,
            rot=rot,
            data=data,
            smearing=smearing,
        )
    elif i == 1:
        exe_ml_plot(
            network=network,
            stencils=stencils,
            layers=layers,
            activation=activation,
            epochs=epochs,
            learning_rate=learning_rate,
            neg=neg,
            angle=angle,
            rot=rot,
            data=data,
            smearing=smearing,
        )
    # '''
    '''
    # 5x5
    network = ['auto']
    stencils = [[5, 5]]
    layers = [[6, 3, 20], [20, 10, 80]]
    activation = ['relu']
    if i == 0:
        exe_ml(
            network=network,
            stencils=stencils,
            layers=layers,
            activation=activation,
            epochs=epochs,
            learning_rate=learning_rate,
            neg=neg,
            angle=angle,
            rot=rot,
            data=data,
            smearing=smearing,
        )
    elif i == 1:
        exe_ml_plot(
            network=network,
            stencils=stencils,
            layers=layers,
            activation=activation,
            epochs=epochs,
            learning_rate=learning_rate,
            neg=neg,
            angle=angle,
            rot=rot,
            data=data,
            smearing=smearing,
        )
    # '''
