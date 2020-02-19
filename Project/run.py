from src.execution import exe_ml, exe_ml_plot
# from src.execution import exe_dg

''' Data Generation '''
'''
stencils = [[3, 3], [5, 5], [7, 3], [3, 7]]
ek = [True, False]
exe_dg(stencils=stencils, ek=ek)
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
epochs = [100]
stencils = [[5, 5]]
activation = ['relu']
learning_rate = [1e-5, 1e-4]
# 1.: Train, 2.: Plot
for i in range(2):
    '''
    # MLP
    network = ['mlp']
    layers = [[100, 80], [100, 80, 50]]
    if i == 0:
        exe_ml(
            network=network,
            stencils=stencils,
            layers=layers,
            activation=activation,
            epochs=epochs,
            learning_rate=learning_rate
        )
    elif i == 1:
        exe_ml_plot(
            network=network,
            stencils=stencils,
            layers=layers,
            activation=activation,
            epochs=epochs,
            learning_rate=learning_rate
        )
    # '''

    '''
    # CVN
    network = ['cvn']
    layers = [[32]]
    if i == 0:
        exe_ml(
            network=network,
            stencils=stencils,
            layers=layers,
            activation=activation,
            epochs=epochs,
            learning_rate=learning_rate
        )
    elif i == 1:
        exe_ml_plot(
            network=network,
            stencils=stencils,
            layers=layers,
            activation=activation,
            epochs=epochs,
            learning_rate=learning_rate
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
            learning_rate=learning_rate
        )
    elif i == 1:
        exe_ml_plot(
            network=network,
            stencils=stencils,
            layers=layers,
            activation=activation,
            epochs=epochs,
            learning_rate=learning_rate
        )
    # '''
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
            learning_rate=learning_rate
        )
    elif i == 1:
        exe_ml_plot(
            network=network,
            stencils=stencils,
            layers=layers,
            activation=activation,
            epochs=epochs,
            learning_rate=learning_rate
        )
    # '''
