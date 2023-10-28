from utils_public import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


grids = load_grids() # Helper function we have provided to load the grids from the dataset
ratings = np.load("datasets/scores.npy") # Load advisor scores
score_order = ["Wellness", "Tax", "Transportation", "Business"] #This is the order of the scores in the dataset
ratings_df = pd.DataFrame(ratings, columns = score_order) #Create a dataframe

advisor_val = 2
grids_subset, ratings_subset = select_rated_subset(grids, ratings[:,advisor_val]) #gets subset of the dataset rated by advisor 2

mask = ratings_subset > 0.85

grids2 = grids_subset[mask]
ratings2 = ratings_subset[mask]

np.savez('grids_advisor2_good.npz', grids2)
np.save('ratings2.npy', ratings2)

# Load the '.npz' file
loaded_data = np.load('grids_advisor2_good.npz')

# Access the arrays from the loaded data
grids_read = loaded_data['arr_0']  # 'arr_0' is the default key used by np.savez

# Load the '.npy' file
ratings_read = np.load('ratings2.npy')

print(grids_read.shape)
print(ratings_read.shape)