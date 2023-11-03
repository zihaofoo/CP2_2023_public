from utils_public import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from CNN_NIC_sub_v2 import *
from scipy.ndimage import rotate
from tensorflow.keras.models import load_model
import pdb

seed_number = 42
np.random.seed(seed_number)
tf.random.set_seed(seed_number)
tf.keras.utils.set_random_seed(seed_number)

# grids = load_grids() # Helper function we have provided to load the grids from the dataset
# ratings = np.load("datasets/scores.npy") # Load advisor scores
# score_order = ["Wellness", "Tax", "Transportation", "Business"] #This is the order of the scores in the dataset
# ratings_df = pd.DataFrame(ratings, columns = score_order) #Create a dataframe
# 
# model0 = get_trained_model(advisor_val = 0, eval_mode = True, seed_number = seed_number)
# model0.save("model0.h5")
# 
# model1 = get_trained_model(advisor_val = 1, eval_mode = True, seed_number = seed_number)
# model1.save("model1.h5")
# 
# model2 = get_trained_model(advisor_val = 2, eval_mode = True, seed_number = seed_number)
# model2.save("model2.h5")
# 
# model3 = get_trained_model(advisor_val = 3, eval_mode = True, seed_number = seed_number)
# model3.save("model3.h5")

# Load the '.npz' file
# loaded_data = np.load('grids_advisor2_good.npz')
# Access the arrays from the loaded data
# grids = loaded_data['arr_0']  # 'arr_0' is the default key used by np.savez

features0 = []
features1 = []
features2 = []
features3 = []
# 
# for grid in grids:
#     features = compute_features(grid, advisor = 0)
#     features = np.nan_to_num(features, nan = 0)
#     features0.append(features)
# features0 = np.array(features0)
# features0[np.isnan(features0)] = 0
# features0 = features0.astype(np.float64)
# np.save('features0.npy', features0)
# 
# for grid in grids:
#     features = compute_features(grid, advisor = 1)
#     features = np.nan_to_num(features, nan = 0)
#     features1.append(features)
# features1 = np.array(features1)
# features1[np.isnan(features1)] = 0
# features1 = features1.astype(np.float64)
# np.save('features1.npy', features1)
# 
# for grid in grids:
#     features = compute_features(grid, advisor = 2)
#     features = np.nan_to_num(features, nan = 0)
#     features2.append(features)
# features2 = np.array(features2)
# features2[np.isnan(features2)] = 0
# features2 = features2.astype(np.float64)
# np.save('features2.npy', features2)
# 
# for grid in grids:
#     features = compute_features(grid, advisor = 3)
#     features = np.nan_to_num(features, nan = 0)
#     features3.append(features)
# features3 = np.array(features3)
# features3[np.isnan(features3)] = 0   
# features3 = features3.astype(np.float64)
# np.save('features3.npy', features3)
# 
# grids_onehot = np.array([one_hot_encode(grid) for grid in grids])
# grids_onehot = grids_onehot.astype(np.float64)
# np.savez('grids_onehot.npz', grids_onehot)

# features0 = np.load('features0.npy')
# features1 = np.load('features1.npy')
# features2 = np.load('features2.npy')
# features3 = np.load('features3.npy')
# 
# with np.load('grids_onehot.npz') as data:
#     grids_onehot = data['arr_0']
# # 
# model0 = load_model("model0.h5")
# model1 = load_model("model1.h5")
# model2 = load_model("model2.h5")
# model3 = load_model("model3.h5")
# # 
# # 
# preds0 = model0.predict([grids_onehot, features0])
# preds1 = model1.predict([grids_onehot, features1])
# preds2 = model2.predict([grids_onehot, features2])
# preds3 = model3.predict([grids_onehot, features3])
# 
preds0 = np.load('preds0.npy')
preds1 = np.load('preds1.npy')
preds2 = np.load('preds2.npy')
preds3 = np.load('preds3.npy')

# 
threshold = 0.85
mask0 = preds0 > threshold 
mask1 = preds1 > threshold 
mask2 = np.logical_and(preds2 > 0.82, preds2 < 0.85)
mask3 = preds3 > threshold 
mask_total = np.logical_and(mask0, np.logical_and(mask1, np.logical_and(mask2, mask3)))


# np.save('mask0.npy', mask0)
# np.save('mask1.npy', mask1)
# np.save('mask2.npy', mask2)
# np.save('mask3.npy', mask3)
# 

# mask0 = np.load('mask0.npy')
# mask1 = np.load('mask1.npy')
# mask2 = np.load('mask2.npy')
# mask3 = np.load('mask3.npy')



grids = load_grids() # Helper function we have provided to load the grids from the dataset
# loaded_data = np.load('grids_advisor2_good.npz')
# Access the arrays from the loaded data
# grids = loaded_data['arr_0']  # 'arr_0' is the default key used by np.savez

mask_total = mask_total.reshape(len(mask_total))
#print(mask_total.shape)
#print(grids.shape)

print(grids[mask_total].shape)
np.savez('grids_filtered_advisor_top_8_below_0.85.npz', grids[mask_total])
all_predictions = grids[mask_total]
top_eight = all_predictions[0:8,:,:].astype(int)
print(top_eight.shape)
final_predictions = np.zeros((4,7,7))


for i in (5,7,9,11,13,15,17,19):
    if i == 5:
        for j in range(i):
            final_predictions = np.vstack((final_predictions,top_eight[0:1]))
    if i == 7:
        for j in range(i):
            final_predictions = np.vstack((final_predictions,top_eight[1:2]))
    if i == 9:
        for j in range(i):
            final_predictions = np.vstack((final_predictions,top_eight[2:3]))
    if i == 11:
        for j in range(i):
            final_predictions = np.vstack((final_predictions,top_eight[3:4]))
    if i == 13:
        for j in range(i):
            final_predictions = np.vstack((final_predictions,top_eight[4:5]))
    if i == 15:
        for j in range(i):
            final_predictions = np.vstack((final_predictions,top_eight[5:6]))
    if i == 17:
        for j in range(i):
            final_predictions = np.vstack((final_predictions,top_eight[6:7]))
    if i == 19:
        for j in range(i):
            final_predictions = np.vstack((final_predictions,top_eight[7:8]))


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
np.save(f"185858285.npy", final_submission)

