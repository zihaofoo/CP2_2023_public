from utils_public import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from CNN_NIC_sub_v2 import *
from scipy.ndimage import rotate
# from tensorflow.keras.models import load_model
import pdb

#original_grids = np.load('6_confirm_good_185858285.npy')
# original_grids = np.vstack((original_grids,np.load('Day_0_assumed_good.npy')))
#original_grids = np.vstack((original_grids,np.load('5_confirm_good_185857885.npy')))
original_grids = np.load('Day2_good.npy')
original_grids = np.vstack((original_grids,np.load('Day0_good.npy'))) # replaces day 0 assume good
original_grids = np.vstack((original_grids,np.load('Day3_good.npy'))) # transformation of 6_confirm_good, day 0 assume good and 5 confirm good. minus the last 8
final_submission = np.vstack((original_grids,original_grids[24:47])).astype(int)
print(final_submission.shape)
print(diversity_score(final_submission))

# np.save('grids_best_v2.npy', original_grids)

# Initialize an empty array to store the 80 matrices
# result_matrices = np.empty((8 * 13, 7, 7), dtype=int)
# 
# # Iterate through the first 10 matrices
# for i in range(13):
#     # Original matrix
#     original_matrix = original_grids[i]
# 
#     # Rotate 90 degrees four times (0, 90, 180, 270 degrees)
#     for j in range(4):
#         result_matrices[i * 4 + j] = np.rot90(original_matrix, j)
# 
# 
#     # Flip horizontally and rotate 90 degrees four times (0, 90, 180, 270 degrees)
#     flipped_matrix = np.fliplr(original_matrix)
#     for j in range(4):
#         result_matrices[4 * 13 + i * 4 + j] = np.rot90(flipped_matrix, j)
# 
# 
# final_submission = result_matrices[:100].astype(int)
# # print(diversity_score(final_submission))
# # for i in range(100):
# #     print(final_submission[i])
# # pdb.set_trace()
## assert final_submission.shape == (100, 7, 7)
## assert final_submission.dtype == int
## assert np.all(np.greater_equal(final_submission, 0) & np.less_equal(final_submission, 4))
## id = np.random.randint(1e8, 1e9-1)
## np.save(f"999888777.npy", final_submission)