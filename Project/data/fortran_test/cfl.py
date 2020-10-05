import regex as re
import numpy as np
import matplotlib.pyplot as plt
import os

# paths = ['2008191431 ml cfl 128', 'FASTEST_3']
# paths = ['2008201426 ml 128 cfl', '2008201426 ml 256 cfl']
# paths = ['FASTEST_1', 'FASTEST_2', 'FASTEST_3', 'FASTEST_4']
paths = ['2008240955 rising Bubble 80x160 0.001', '2008240955 rising Bubble 80x160 0.01', '2008240955 rising Bubble 80x160 0.02']
for path in paths:
    file_name = os.path.join(path, 'output.txt')
    with open(file_name, 'r') as myfile:
        data = myfile.read()
        variables = np.array(re.findall(r'(?<=cfl\s*\=\s*)(\d\.\d+)(?=\s*dif)', data), dtype=np.float32)
        print(f'CFL max:\t{variables.max():.3f}')
        plt.plot(variables, label=path)
plt.legend()
plt.show()
