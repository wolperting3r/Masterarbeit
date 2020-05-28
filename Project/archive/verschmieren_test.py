import numpy as np

b = np.array([[1, 1, 1, 0.8, 0], [1, 1, 1, 0.6, 0], [1, 1, 0.4, 0, 0], [0.2, 0.2, 0, 0, 0], [0, 0, 0, 0, 0]])
b = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0.2, 0.2], [0, 0, 0.4, 1, 1], [0, 0.6, 1, 1, 1], [0, 0.8, 1, 1, 1]])

b = np.round(b, 3)

be = np.zeros(b.shape)
bn = np.zeros(b.shape)
bp = np.zeros(b.shape)
bm = np.zeros(b.shape)

fx = 0.5
fy = 0.5
for i in range(b.shape[0]-1):
    for j in range(b.shape[1]-1):
        be[i, j] = b[i, j]*(1-fx) + b[i, j+1]*fx
        bn[i, j] = b[i, j]*(1-fy) + b[i+1, j]*fy

bn = np.round(bn, 3)
be = np.round(be, 3)
print(f'b:\n{b}')
print(f'be:\n{be}')
print(f'bn:\n{bn}')


SE = 1
SN = 1
for i in range(b.shape[0]-1):
    for j in range(b.shape[1]-1):
        bm[i, j] = bm[i, j] + bn[i-1, j]*SE]
        bm[i, j] = bm[i, j] + be[i, j]*SE + bn[i, j]*SN
        bm[i+1, j] = bm[i+1, j] + bn[i, j]*SE
        bm[i, j+1] = bm[i, j+1] + be[i, j]*SN
        
        bp[i, j] = bp[i, j] + SE + SN
        bp[i+1, j] = bp[i+1, j] + SN
        bp[i, j+1] = bp[i, j+1] + SE

bm = np.round(bm, 3)
bp = np.round(bp, 3)

print(f'bm:\n{bm}')
print(f'bp:\n{bp}')

bm = np.round(bm/bp, 3)
print(f'bm:\n{bm}')
