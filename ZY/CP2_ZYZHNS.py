from utils_public import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from autogluon.tabular import TabularPredictor
from datetime import datetime
import pdb 
import random
from scipy.spatial import distance
from CP2_sub import *


## Implementing ML model 
grids = load_grids()                                            # Helper function we have provided to load the grids from the dataset
ratings = np.load("datasets/scores.npy")                        # Load advisor scores
score_order = ["Wellness", "Tax", "Transportation", "Business"] # This is the order of the scores in the dataset
ratings_df = pd.DataFrame(ratings, columns = score_order)       # Create a dataframe

all_predictions = []
all_predictors = []

for i in range(2,3):
    predictions, predictor = fit_plot_predict_zyzh(grids, ratings, i)
#     all_predictions.append(predictions)
#     all_predictors.append(predictor)
# 
# final_prediction_array = np.stack(all_predictions).T
# min_predictions = np.min(final_prediction_array, axis = 1)
# top_100_indices = np.argpartition(min_predictions, -100)[-100:] # indices of top 100 designs (as sorted by minimum advisor score)
# final_submission = grids[top_100_indices].astype(int)
# 
# assert final_submission.shape == (100, 7, 7)
# assert final_submission.dtype == int
# assert np.all(np.greater_equal(final_submission, 0) & np.less_equal(final_submission, 4))
# id = np.random.randint(1e8, 1e9-1)
# np.save(f"{id}.npy", final_submission)
# 
# 
