from utils_public import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from CNN_NIC_sub import *

np.random.seed(42)
tf.random.set_seed(42)
grids = load_grids() # Helper function we have provided to load the grids from the dataset
ratings = np.load("datasets/scores.npy") # Load advisor scores
score_order = ["Wellness", "Tax", "Transportation", "Business"] #This is the order of the scores in the dataset
ratings_df = pd.DataFrame(ratings, columns = score_order) #Create a dataframe

advisor_val = 2
grids_subset, ratings_subset = select_rated_subset(grids, ratings[:,advisor_val]) #gets subset of the dataset rated by advisor 2
grids_encoded = np.array([one_hot_encode(grid) for grid in grids_subset])

# First split: 80% for training, 20% for temp (to be divided into test and validation)
grids_train, grids_test, ratings_train, ratings_test = train_test_split(grids_encoded, ratings_subset, test_size = 0.2, random_state = 42)

# features_subset = parallel_compute(grids_subset, advisor_val)
features_subset = []
for grid in grids_subset:
    features = compute_features(grid, advisor = advisor_val)
    features = np.nan_to_num(features, nan = 0)
    features_subset.append(features)
features_subset = np.array(features_subset)
features_subset[np.isnan(features_subset)] = 0
features_train, features_test, ratings_train, ratings_test = train_test_split(features_subset, ratings_subset, test_size = 0.2, random_state = 42)

grids_train = grids_train.astype(np.float32)
features_train = features_train.astype(np.float32)
grids_test = grids_test.astype(np.float32)
features_test = features_test.astype(np.float32)

model = create_combined_model()
model.summary()

batch_size = 64
epochs = 29

model.fit([grids_train, features_train], ratings_train, epochs=epochs, batch_size=batch_size)
preds_train = model.predict([grids_train, features_train])
preds_test = model.predict([grids_test, features_test])
plot_and_r2(preds_train, preds_test, ratings_train, ratings_test, advisor_val)
