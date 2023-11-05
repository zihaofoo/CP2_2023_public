import numpy as np
import pandas as pd

import pdb

# confirm_good = np.empty((11,7,7))

# 302828280 - 19 valid (1,2,16)             -->0, 1, 4
# 302828281 - 50 valid (2,16,32)            -->1, 4, 5
# 302828282 - 95 valid (2,4,8,16,32,33)     -->1, 2, 3, 4, 5, 6
# 302828283 - 96 valid (1,2,4,8,16,32,33)   -->0, 1, 2, 3, 4, 5, 6


grids = np.load('grids_302828280.npy')

confirm_good = grids[0:1,:,:]
for i in (1, 4):
    confirm_good = np.vstack((confirm_good,grids[i:i+1,:,:]))



grids = np.load('grids_302828281.npy')
for i in (1, 4, 5):
    confirm_good = np.vstack((confirm_good,grids[i:i+1,:,:]))

print(confirm_good.shape)

grids = np.load('grids_302828282.npy')
for i in (1, 2, 3, 4, 5, 6):
    confirm_good = np.vstack((confirm_good,grids[i:i+1,:,:]))

grids = np.load('grids_302828283.npy')
for i in (0, 1, 2, 3, 4, 5, 6):
    confirm_good = np.vstack((confirm_good,grids[i:i+1,:,:]))

print(confirm_good.shape)

np.save('Day0_good.npy', confirm_good)
