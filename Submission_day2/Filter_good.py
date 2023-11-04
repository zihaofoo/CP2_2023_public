import numpy as np
import pandas as pd

import pdb

# confirm_good = np.empty((11,7,7))


# 206350780 - 1 valid  (1)              -->0
# 206350781 - 0 valid                   -->
# 206350782 - 48 valid (32,16)          -->4, 5
# 206350783 - 0 valid                   -->
# 206350784 - 16 valid (16)             -->4
# 206350785 - 0 valid                   -->
# 206350786 - 19 valid (16, 2, 1)       -->0,1,4
# 206350787 - 20 valid (16, 4)          -->2,4
# 206350788 - 5 valid (4, 1)            -->0,2

grids = np.load('grids_206350780.npy')
for i in (0,):
    confirm_good = grids[i:i+1,:,:]



grids = np.load('grids_206350782.npy')
for i in (4,5):
    confirm_good = np.vstack((confirm_good,grids[i:i+1,:,:]))

print(confirm_good.shape)

grids = np.load('grids_206350784.npy')
for i in (4,):
    confirm_good = np.vstack((confirm_good,grids[i:i+1,:,:]))

grids = np.load('grids_206350786.npy')
for i in (0,1,4):
    confirm_good = np.vstack((confirm_good,grids[i:i+1,:,:]))

grids = np.load('grids_206350787.npy')
for i in (2,4):
    confirm_good = np.vstack((confirm_good,grids[i:i+1,:,:]))

grids = np.load('grids_206350788.npy')
for i in (0,2):
    confirm_good = np.vstack((confirm_good,grids[i:i+1,:,:]))

print(confirm_good.shape)

np.save('Day2_good.npy', confirm_good)