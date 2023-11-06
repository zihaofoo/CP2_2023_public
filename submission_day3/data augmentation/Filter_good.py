import numpy as np
import pandas as pd

import pdb

# confirm_good = np.empty((11,7,7))


# 328820000 - 0 valid
# 328820001 - 0 valid
# 328820002 - 0 valid
# 328820003 - 0 valid
# 328820004 - 96 valid (1,2,4,8,16,32,33)   -->0, 1, 2, 3, 4, 5, 6
# 328820005 - 96 valid (1,2,4,8,16,32,33)   -->0, 1, 2, 3, 4, 5, 6
# 328820006 - 96 valid (1,2,4,8,16,32,33)   -->0, 1, 2, 3, 4, 5, 6
# 328820007 - 0 valid
# 328820008 - 0 valid
# 328820009 - 84 valid (1,2,16,32,33)       -->0, 1, 4, 5, 6
# 328820010 - 0 valid
# 328820011 - 0 valid
# 328820012 - 96 valid (1,2,4,8,16,32,33)   -->0, 1, 2, 3, 4, 5, 6
# 328820013 - 0 valid
# 328820014 - 0 valid
# 328820015 - 0 valid
# 328820016 - 96 valid (1,2,4,8,16,32,33)   -->0, 1, 2, 3, 4, 5, 6
# 328820017 - 0 valid
# 328820018 - 96 valid (1,2,4,8,16,32,33)   -->0, 1, 2, 3, 4, 5, 6
# 328820019 - 0 valid

grids = np.load('grids_328820004.npy')
confirm_good = grids[0:1,:,:]
for i in (1, 2, 3, 4, 5, 6):
    confirm_good = np.vstack((confirm_good,grids[i:i+1,:,:]))
print(confirm_good.shape)

grids = np.load('grids_328820005.npy')
for i in (0, 1, 2, 3, 4, 5, 6):
    confirm_good = np.vstack((confirm_good,grids[i:i+1,:,:]))
print(confirm_good.shape)

grids = np.load('grids_328820006.npy')
for i in (0, 1, 2, 3, 4, 5, 6):
    confirm_good = np.vstack((confirm_good,grids[i:i+1,:,:]))
print(confirm_good.shape)

grids = np.load('grids_328820009.npy')
for i in (0, 1, 4, 5, 6):
    confirm_good = np.vstack((confirm_good,grids[i:i+1,:,:]))
print(confirm_good.shape)

grids = np.load('grids_3288200012.npy')
for i in (0, 1, 2, 3, 4, 5, 6):
    confirm_good = np.vstack((confirm_good,grids[i:i+1,:,:]))
print(confirm_good.shape)

grids = np.load('grids_3288200016.npy')
for i in (0, 1, 2, 3, 4, 5, 6):
    confirm_good = np.vstack((confirm_good,grids[i:i+1,:,:]))
print(confirm_good.shape)

grids = np.load('grids_3288200018.npy')
for i in (0, 1, 2, 3, 4, 5, 6):
    confirm_good = np.vstack((confirm_good,grids[i:i+1,:,:]))
print(confirm_good.shape)

np.save('Day3_good.npy', confirm_good)
print(confirm_good.shape)