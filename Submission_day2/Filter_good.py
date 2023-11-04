import numpy as np
import pandas as pd

import pdb




#print(mask_total.shape)
#print(grids.shape)
grids = np.load('grids_206350780.npy')
submit = np.load('206350780.npy')
print(grids)
pdb.set_trace()
confirm_good = np.empty((6,7,7))
print(confirm_good)
print('---------------------------------------')
# confirm_good = np.vstack((np.vstack((top_eight[0:1],top_eight[2:3])),top_eight[5:]))
# np.save(f"5_confirm_good_185857885.npy", confirm_good)
# print(confirm_good.shape)

# confirm 35 is invalid. 4 are all zeros. meaning 3 sets are invalid (diversity score without 7, 11, 13 is the closest)
for i in (0,1,2,3,4,5):
    confirm_good = np.vstack(grids[i])
    print(confirm_good)
