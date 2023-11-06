import numpy as np
import pandas as pd

import pdb

# confirm_good = np.empty((11,7,7))



# 4111111110 - 58 valid, compared with results of 82-78 day1, 1,4,37 are invalid. 2,8,16,32 are valid  --> 1, 3, 4, 5 ---> 1,3 are seen in grids1
# 4111111111 - 41111110[0] is the same as 411111111[3]   --> 1,4,16,37 --> 0, 2, 4, 6 ---> 0,2 are seen in grids 2
# 4111111112 - 80 valid (1,2,8,32,33)   -->0, 1, 3, 5


grids = np.load('grids_411111110.npy')
confirm_good = grids[4:5,:,:]
for i in (5,):
    confirm_good = np.vstack((confirm_good,grids[i:i+1,:,:]))
print(confirm_good.shape)

grids = np.load('grids_411111111.npy')
for i in (4, 6):
    confirm_good = np.vstack((confirm_good,grids[i:i+1,:,:]))
print(confirm_good.shape)

grids = np.load('grids_411111112.npy')
for i in (0, 1, 3, 5):
    confirm_good = np.vstack((confirm_good,grids[i:i+1,:,:]))
print(confirm_good.shape)

grids0 = np.load('grids_411111110.npy')
grids1 = np.load('grids_411111111.npy')
grids2 = np.load('grids_411111112.npy')



np.save('Day4_good.npy', confirm_good)
print(confirm_good.shape)