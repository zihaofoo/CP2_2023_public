from utils_public import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from CNN_NIC_sub_v2 import *
from scipy.ndimage import rotate
from tensorflow.keras.models import load_model

seed_number = 42
np.random.seed(seed_number)
tf.random.set_seed(seed_number)
tf.keras.utils.set_random_seed(seed_number)

grids = load_grids() # Helper function we have provided to load the grids from the dataset
ratings = np.load("datasets/scores.npy") # Load advisor scores
score_order = ["Wellness", "Tax", "Transportation", "Business"] #This is the order of the scores in the dataset
ratings_df = pd.DataFrame(ratings, columns = score_order) #Create a dataframe

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

## for grid in grids:
##     features = compute_features(grid, advisor = 0)
##     features = np.nan_to_num(features, nan = 0)
##     features0.append(features)
## 
##     features = compute_features(grid, advisor = 1)
##     features = np.nan_to_num(features, nan = 0)
##     features1.append(features)
## 
##     features = compute_features(grid, advisor = 2)
##     features = np.nan_to_num(features, nan = 0)
##     features2.append(features)
## 
##     features = compute_features(grid, advisor = 3)
##     features = np.nan_to_num(features, nan = 0)
##     features3.append(features)
## 
## features0 = np.array(features0)
## features0[np.isnan(features0)] = 0
## features1 = np.array(features1)
## features1[np.isnan(features1)] = 0
## features2 = np.array(features2)
## features2[np.isnan(features2)] = 0
## features3 = np.array(features3)
## features3[np.isnan(features3)] = 0   
## 
## 
## grids_onehot = np.array([one_hot_encode(grid) for grid in grids])
## grids_onehot = grids_onehot.astype(np.float64)
## features0 = features0.astype(np.float64)
## features1 = features1.astype(np.float64)
## features2 = features2.astype(np.float64)
## features3 = features3.astype(np.float64)
## 
# np.save('features0.npy', features0)
# np.save('features1.npy', features1)
# np.save('features2.npy', features2)
# np.save('features3.npy', features3)
# np.savez('grids_onehot.npz', grids_onehot)

features0 = np.load('features0.npy')
features1 = np.load('features1.npy')
features2 = np.load('features2.npy')
features3 = np.load('features3.npy')

with np.load('grids_onehot.npz') as data:
    grids_onehot = data['arr_0']

model0 = load_model("model0.h5")
model1 = load_model("model1.h5")
model2 = load_model("model2.h5")
model3 = load_model("model3.h5")


preds0 = model0.predict([grids_onehot, features0])
preds1 = model1.predict([grids_onehot, features1])
preds2 = model2.predict([grids_onehot, features2])
preds3 = model3.predict([grids_onehot, features3])

threshold = 0.85
mask0 = preds0 > threshold 
mask1 = preds1 > threshold 
mask2 = preds2 > threshold 
mask3 = preds3 > threshold 
mask_total = np.logical_and(mask0, np.logical_and(mask1, mask3))

grids = load_grids() # Helper function we have provided to load the grids from the dataset
# loaded_data = np.load('grids_advisor2_good.npz')
# Access the arrays from the loaded data
# grids = loaded_data['arr_0']  # 'arr_0' is the default key used by np.savez
mask_total = mask_total.reshape(len(mask_total))
print(mask_total.shape)
print(grids.shape)

print(grids[mask_total].shape)
np.savez('grids_filtered.npz', grids[mask_total])

all_predictions = grids[mask_total]

final_prediction_array = np.stack(all_predictions).T
min_predictions = np.min(final_prediction_array, axis = 1)
top_100_indices = np.argpartition(min_predictions, -100)[-100:] # indices of top 100 designs (as sorted by minimum advisor score)
final_submission = grids[top_100_indices].astype(int)

assert final_submission.shape == (100, 7, 7)
assert final_submission.dtype == int
assert np.all(np.greater_equal(final_submission, 0) & np.less_equal(final_submission, 4))
id = np.random.randint(1e8, 1e9-1)
np.save(f"{id}.npy", final_submission)

