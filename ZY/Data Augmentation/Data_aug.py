from utils_public import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from CNN_NIC_sub_v2 import *
from scipy.ndimage import rotate
# from tensorflow.keras.models import load_model
import pdb

original_grids = np.load('6_confirm_good_185858285.npy')
original_grids = np.vstack((original_grids,np.load('grids_filtered_advisor_pass_all.npy')))
original_grids = np.vstack((original_grids,np.load('5_confirm_good_185857885.npy')))

print(original_grids.shape[0])

# Initialize an empty array to store the 80 matrices
result_matrices = np.empty((8 * original_grids.shape[0], 7, 7), dtype=int)

# Iterate through the first 10 matrices
for i in range(original_grids.shape[0]):
    # Original matrix
    original_matrix = original_grids[i]

    # Rotate 90 degrees four times (0, 90, 180, 270 degrees)
    for j in range(4):
        result_matrices[i * 4 + j] = np.rot90(original_matrix, j)


    # Flip horizontally and rotate 90 degrees four times (0, 90, 180, 270 degrees)
    flipped_matrix = np.fliplr(original_matrix)
    for j in range(4):
        result_matrices[4 * original_grids.shape[0] + i * 4 + j] = np.rot90(flipped_matrix, j)

print(result_matrices.shape)
np.save('grids_augmented_day1.npy', result_matrices)