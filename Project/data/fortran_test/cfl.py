import regex as re
import numpy as np
import matplotlib.pyplot as plt
import os

# paths = ['2008191431 ml cfl 128', 'FASTEST_3']
paths = ['FASTEST_2', 'FASTEST_3']
for path in paths:
    file_name = os.path.join(path, 'output.txt')
    with open(file_name, 'r') as myfile:
        data = myfile.read()
        variables = np.array(re.findall(r'(?<=cfl\s*\=\s*)(\d\.\d+)(?=\s*dif)', data), dtype=np.float32)
        print(f'CFL max:\t{variables.max():.3f}')
        plt.plot(variables, label=path)
plt.legend()
plt.show()
