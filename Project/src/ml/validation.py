import matplotlib.pyplot as plt
import os
import io
import sys
from contextlib import redirect_stdout


def validate_model_loss(model, train_data, train_labels, test_data, test_labels, parameters):
    # Print MSE and MAE
    path = os.path.dirname(os.path.abspath(sys.argv[0]))
    param_str = parameters['filename']
    file_name = os.path.join(path, 'models', 'logs', 'log' + param_str + '.txt')
    # Catch print output of tensorflow functions
    f = io.StringIO()
    with redirect_stdout(f):
        print(str(model.evaluate(train_data, train_labels, verbose=2)))
        print('\n')
        print(str(model.evaluate(test_data, test_labels, verbose=2)))
        print('\n')
        print(str(model.summary()))
    out = f.getvalue()
    # Write output into logfile
    with open(file_name, 'w') as logfile:
        logfile.write(out)
        # logfile.write('\n')


def validate_model_plot(model, test_data, test_labels, parameters):
    # Get predictions for test data
    test_predictions = model.predict(test_data, batch_size=parameters['batch_size']).flatten()

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Create scatterplot test_predictions vs test_labels
    ax.scatter(test_labels, test_predictions, alpha=0.05)
    ax.set_xlabel('True Values [MPG]')
    ax.set_ylabel('Predictions [MPG]')
    # lims = [min(test_labels), max(test_labels)]
    lims = [-0.2, 4/3+0.2]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims, lims)

    # Save plot
    path = os.path.dirname(os.path.abspath(sys.argv[0]))
    param_str = parameters['filename']
    file_name = os.path.join(path, 'models', 'figures', 'fig' + param_str + '.png')
    fig.tight_layout()
    fig.savefig(file_name, dpi=150)
    # plt.show()
    plt.close()
