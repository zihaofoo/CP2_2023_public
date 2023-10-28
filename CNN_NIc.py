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

advisor_val = 2
grids_subset, ratings_subset = select_rated_subset(grids, ratings[:,advisor_val]) #gets subset of the dataset rated by advisor 2
# rotated_array1 = rotate(grids_subset, angle=90, axes=(1,2), reshape=True)
# rotated_array2 = rotate(grids_subset, angle=180, axes=(1,2), reshape=True)
# rotated_array3 = rotate(grids_subset, angle=270, axes=(1,2), reshape=True)

# grids_subset = np.vstack((grids_subset, rotated_array1, rotated_array2, rotated_array3))
# ratings_subset = np.concatenate((ratings_subset, ratings_subset, ratings_subset, ratings_subset))

# First split: 80% for training, 20% for temp (to be divided into test and validation)
grids_train, grids_test, ratings_train, ratings_test = train_test_split(grids_subset, ratings_subset, test_size = 0.2, random_state = 42)

rotated_array1 = rotate(grids_train, angle=90, axes=(1,2), reshape=True)
# rotated_array2 = rotate(grids_train, angle=180, axes=(1,2), reshape=True)
# rotated_array3 = rotate(grids_train, angle=270, axes=(1,2), reshape=True)
grids_train = np.vstack((grids_train, rotated_array1))
ratings_train = np.concatenate((ratings_train, ratings_train))

# features_subset = parallel_compute(grids_subset, advisor_val)
features_train = []
for grid in grids_train:
    features = compute_features(grid, advisor = advisor_val)
    features = np.nan_to_num(features, nan = 0)
    features_train.append(features)
features_train = np.array(features_train)
features_train[np.isnan(features_train)] = 0

features_test = []
for grid in grids_test:
    features = compute_features(grid, advisor = advisor_val)
    features = np.nan_to_num(features, nan = 0)
    features_test.append(features)
features_test = np.array(features_test)
features_test[np.isnan(features_test)] = 0

grids_train = np.array([one_hot_encode(grid) for grid in grids_train])
grids_test = np.array([one_hot_encode(grid) for grid in grids_test])

grids_train = grids_train.astype(np.float64)
features_train = features_train.astype(np.float64)
grids_test = grids_test.astype(np.float64)
features_test = features_test.astype(np.float64)

model = create_combined_model(advisor = advisor_val)
model.summary()

batch_size = 64
epochs = 15

model.fit([grids_train, features_train], ratings_train, epochs=epochs, batch_size=batch_size)
preds_train = model.predict([grids_train, features_train])
preds_test = model.predict([grids_test, features_test])
plot_and_r2(preds_train, preds_test, ratings_train, ratings_test, advisor_val)
