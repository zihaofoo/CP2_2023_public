# from utils_public import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from CNN_NIC_sub_v2 import *
from scipy.ndimage import rotate
# from tensorflow.keras.models import load_model
import pdb


all_predictions = np.load('Day_0_assumed_good.npy')
print(all_predictions.shape)

for i in range(4):


    top_six = all_predictions[i * 7 : i * 7 + 7 ,:,:].astype(int)
    
    final_predictions = np.zeros((4,7,7))


    for j in range(6):
        for k in range(2 ** j):
           final_predictions = np.vstack((final_predictions,top_six[j : j + 1]))
        
        print((np.array(final_predictions)).shape)

    for l in range(33):
        final_predictions = np.vstack((final_predictions,top_six[6:7]))

    print(np.array(final_predictions).shape)
    final_submission = final_predictions.astype(int)

    #--------------------------------------------------------------------------------
    ### this is for creating 100 entries from top 30
    ## # Assuming final_submission is a (28, 7, 7) array
    ## temp = final_submission[:10]
    ## 
    ## # Initialize an empty array to store the 80 matrices
    ## result_matrices = np.empty((80, 7, 7), dtype=int)
    ## 
    ## # Iterate through the first 10 matrices
    ## for i in range(10):
    ##     # Original matrix
    ##     original_matrix = temp[i]
    ## 
    ##     # Rotate 90 degrees four times (0, 90, 180, 270 degrees)
    ##     for j in range(4):
    ##         result_matrices[i * 4 + j] = np.rot90(original_matrix, j)
    ## 
    ##     # Flip horizontally and rotate 90 degrees four times (0, 90, 180, 270 degrees)
    ##     flipped_matrix = np.fliplr(original_matrix)
    ##     for j in range(4):
    ##         result_matrices[40 + i * 4 + j] = np.rot90(flipped_matrix, j)
    ## 
    ## final_submission = np.vstack((final_submission[10:],result_matrices))
    ## print(final_submission.shape)
    #--------------------------------------------------------------------------------

    # Now, result_matrices contains the 80 modified matrices

    # final_prediction_array = np.stack(all_predictions).T
    # min_predictions = np.min(final_prediction_array, axis = 1)
    # top_100_indices = np.argpartition(min_predictions, -100)[-100:] # indices of top 100 designs (as sorted by minimum advisor score)
    # final_submission = grids[top_100_indices].astype(int)
    # pdb.set_trace()



    assert final_submission.shape == (100, 7, 7)
    assert final_submission.dtype == int
    assert np.all(np.greater_equal(final_submission, 0) & np.less_equal(final_submission, 4))
    id = np.random.randint(1e8, 1e9-1)
    filename = f"30282828{i}.npy"
    grids_filename = f"grids_{filename}"
    np.save(grids_filename, top_six)
    np.save(filename, final_submission)

