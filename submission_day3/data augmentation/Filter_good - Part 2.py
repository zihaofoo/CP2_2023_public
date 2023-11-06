import numpy as np
import pandas as pd

import pdb

# confirm_good = np.empty((11,7,7))



# 328820020 - 96 valid (1,2,4,8,16,32,33)   -->0, 1, 2, 3, 4, 5, 6
# 328820021 - 96 valid (1,2,4,8,16,32,33)   -->0, 1, 2, 3, 4, 5, 6
# 328820022 - 96 valid (1,2,4,8,16,32,33)   -->0, 1, 2, 3, 4, 5, 6
# 328820023 - 96 valid (1,2,4,8,16,32,33)   -->0, 1, 2, 3, 4, 5, 6
# 328820024 - 96 valid (1,2,4,8,16,32,33)   -->0, 1, 2, 3, 4, 5, 6
# 328820025 - 96 valid (1,2,4,8,16,32,33)   -->0, 1, 2, 3, 4, 5, 6
# 328820026 - 96 valid (1,2,4,8,16,32,33)   -->0, 1, 2, 3, 4, 5, 6
# 328820027 - 96 valid (1,2,4,8,16,32,33)   -->0, 1, 2, 3, 4, 5, 6



grids = np.load('grids_3288200020.npy')
confirm_good = grids[0:1,:,:]
for i in (1, 2, 3, 4, 5, 6):
    confirm_good = np.vstack((confirm_good,grids[i:i+1,:,:]))
print(confirm_good.shape)

grids = np.load('grids_3288200021.npy')
for i in (0, 1, 2, 3, 4, 5, 6):
    confirm_good = np.vstack((confirm_good,grids[i:i+1,:,:]))
print(confirm_good.shape)

grids = np.load('grids_3288200022.npy')
for i in (0, 1, 2, 3, 4, 5, 6):
    confirm_good = np.vstack((confirm_good,grids[i:i+1,:,:]))
print(confirm_good.shape)

grids = np.load('grids_3288200023.npy')
for i in (0, 1, 2, 3, 4, 5, 6):
    confirm_good = np.vstack((confirm_good,grids[i:i+1,:,:]))
print(confirm_good.shape)

grids = np.load('grids_3288200024.npy')
for i in (0, 1, 2, 3, 4, 5, 6):
    confirm_good = np.vstack((confirm_good,grids[i:i+1,:,:]))
print(confirm_good.shape)

grids = np.load('grids_3288200025.npy')
for i in (0, 1, 2, 3, 4, 5, 6):
    confirm_good = np.vstack((confirm_good,grids[i:i+1,:,:]))
print(confirm_good.shape)

grids = np.load('grids_3288200026.npy')
for i in (0, 1, 2, 3, 4, 5, 6):
    confirm_good = np.vstack((confirm_good,grids[i:i+1,:,:]))
print(confirm_good.shape)

grids = np.load('grids_3288200027.npy')
for i in (0, 1, 2, 3, 4, 5, 6):
    confirm_good = np.vstack((confirm_good,grids[i:i+1,:,:]))
print(confirm_good.shape)
pdb.set_trace()
np.save('Day3_part2_good.npy', confirm_good)
print(confirm_good.shape)