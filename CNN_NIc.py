from utils_public import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from CNN_NIC_sub_v2 import *
from scipy.ndimage import rotate

np.random.seed(42)
tf.random.set_seed(42)
tf.keras.utils.set_random_seed(42)

grids = load_grids() # Helper function we have provided to load the grids from the dataset
ratings = np.load("datasets/scores.npy") # Load advisor scores
score_order = ["Wellness", "Tax", "Transportation", "Business"] #This is the order of the scores in the dataset
ratings_df = pd.DataFrame(ratings, columns = score_order) #Create a dataframe

model0 = get_trained_model(advisor_val = 0)
model1 = get_trained_model(advisor_val = 1)
model2 = get_trained_model(advisor_val = 2)
model3 = get_trained_model(advisor_val = 3)

# Load the '.npz' file
loaded_data = np.load('grids_advisor2_good.npz')
# Access the arrays from the loaded data
grids = loaded_data['arr_0']  # 'arr_0' is the default key used by np.savez

features0 = []
features1 = []
features2 = []
features3 = []

for grid in grids:
    features = compute_features(grid, advisor = 0)
    features = np.nan_to_num(features, nan = 0)
    features0.append(features)

    features = compute_features(grid, advisor = 1)
    features = np.nan_to_num(features, nan = 0)
    features1.append(features)

    features = compute_features(grid, advisor = 2)
    features = np.nan_to_num(features, nan = 0)
    features2.append(features)

    features = compute_features(grid, advisor = 3)
    features = np.nan_to_num(features, nan = 0)
    features3.append(features)

features0 = np.array(features0)
features0[np.isnan(features0)] = 0
features1 = np.array(features1)
features1[np.isnan(features1)] = 0
features2 = np.array(features2)
features2[np.isnan(features2)] = 0
features3 = np.array(features3)
features3[np.isnan(features3)] = 0   

grids = np.array([one_hot_encode(grid) for grid in grids])
grids = grids.astype(np.float64)
features0 = features0.astype(np.float64)
features1 = features1.astype(np.float64)
features2 = features2.astype(np.float64)
features3 = features3.astype(np.float64)

preds0 = model0.predict([grids, features0])
preds1 = model1.predict([grids, features1])
preds2 = model2.predict([grids, features2])
preds3 = model3.predict([grids, features3])

threshold = 0.85
mask0 = preds0 > threshold 
mask1 = preds1 > threshold 
mask2 = preds2 > threshold 
mask3 = preds3 > threshold 

mask_total = np.logical_and(mask0, np.logical_and(mask1, mask3))

np.savez('grids_advisor2_filtered.npz', grids[mask_total])
