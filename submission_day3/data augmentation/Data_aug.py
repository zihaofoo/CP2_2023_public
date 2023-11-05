# from utils_public import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from CNN_NIC_sub_v2 import *
from scipy.ndimage import rotate
# from tensorflow.keras.models import load_model
import pdb

original_grids = np.load('6_confirm_good_185858285.npy')
original_grids = np.vstack((original_grids,np.load('Day_0_assumed_good.npy')))
original_grids = np.vstack((original_grids,np.load('5_confirm_good_185857885.npy')))

print(original_grids.shape[0])


# Iterate through the first 10 matrices
for i in range(28):

    final_predictions = np.zeros((4,7,7))
    # Original matrix
    original_matrix = original_grids[i]
    original_matrix = np.expand_dims(original_matrix,axis=0)
    collect_matrix = original_grids[i]
    collect_matrix = np.expand_dims(collect_matrix,axis=0)
    print(original_matrix)
    for j in range(3):
        for k in range(2 ** j):
           final_predictions = np.vstack((final_predictions,np.rot90(original_matrix, j+1, axes=(1, 2))))
        print(np.rot90(original_matrix, j+1, axes=(1, 2)))
        collect_matrix = np.vstack((collect_matrix,np.rot90(original_matrix, j+1, axes=(1, 2))))

    flipped_matrix = np.fliplr(original_matrix)
    print('--------------------------------------')
    print(flipped_matrix)
    for j in range(3):
        for k in range(2 ** (j + 3)):
           final_predictions = np.vstack((final_predictions,np.rot90(flipped_matrix, j+1, axes=(1, 2))))
        print(np.rot90(flipped_matrix, j+1, axes=(1, 2)))
        collect_matrix = np.vstack((collect_matrix,np.rot90(flipped_matrix, j+1, axes=(1, 2))))

    for l in range(33):
        final_predictions = np.vstack((final_predictions,np.rot90(flipped_matrix, 4, axes=(1, 2))))
        #print(np.rot90(flipped_matrix, 4, axes=(1, 2)))
    collect_matrix = np.vstack((collect_matrix,np.rot90(flipped_matrix, 4, axes=(1, 2))))

    print(final_predictions.shape)
    final_submission = final_predictions.astype(int)
    assert final_submission.shape == (100, 7, 7)
    assert final_submission.dtype == int
    assert np.all(np.greater_equal(final_submission, 0) & np.less_equal(final_submission, 4))
    id = np.random.randint(1e8, 1e9-1)
    filename = f"32882000{i}.npy"
    grids_filename = f"grids_{filename}"
    np.save(grids_filename, collect_matrix)
    np.save(filename, final_submission)
    
# grid = np.load('328820000.npy')
# pdb.set_trace()

## final_submission = final_predictions.astype(int)
## # print(diversity_score(final_submission))
## # for i in range(100):
## #     print(final_submission[i])
## # pdb.set_trace()
## assert final_submission.shape == (100, 7, 7)
## assert final_submission.dtype == int
## assert np.all(np.greater_equal(final_submission, 0) & np.less_equal(final_submission, 4))
## id = np.random.randint(1e8, 1e9-1)
## np.save(f"123.npy", final_submission)