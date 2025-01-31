from utils_public import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from autogluon.tabular import TabularPredictor
from datetime import datetime
import pdb 

## Mics Functions
def plot_and_r2(preds_train, preds_test, ratings_train, ratings_test, i):
    """ Plot of prediction against experiment. """
    fig,ax = plt.subplots(figsize = (6,6))
    ax.scatter(ratings_train, preds_train, label='Train Set Preds', s=3, c = "#F08E18") #Train set in orange
    ax.scatter(ratings_test, preds_test, label='Test Set Preds', s=5, c = "#DC267F") #Test set in magenta
    ax.plot([0,1], [0,1], label="target", linewidth=3, c="k") # Target line in Black
    ax.set_xlabel("Actual Rating")
    ax.set_ylabel("Predicted Rating")
    ax.set_title(f"Advisor {i} Predictions")
    ax.legend(loc = 'lower right')
    ax.text(x = 0, y = 1, s = r"Training R^{2}: " + np.str_(r2_score(ratings_train, preds_train)))
    ax.text(x = 0, y = 0.95, s = r"Testing R^{2}: " + np.str_(r2_score(ratings_test, preds_test)))
    current_datetime = datetime.now()
    current_time = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    fig.savefig('Prediction' + np.str_(i) + '_' + current_time + '.pdf')
    # plt.show()
    print(f"Train Set R2 score: {r2_score(ratings_train, preds_train)}") #Calculate R2 score
    print(f"Test Set R2 score: {r2_score(ratings_test, preds_test)}")

def get_predictions(grids, ratings, predictor):
    """ Function to predict advisor score using a trained predictor."""
    grids_df = pd.DataFrame(grids, columns = range(grids.shape[1]))
    predictions = predictor.predict(grids_df).values
    mask = np.where(~np.isnan(ratings))
    predictions[mask] = ratings[mask]
    return predictions

def fit_plot_predict(grids, ratings, i):
    """ Function to implement autoML to correlate test data and labels, for a given advisor i"""
    grids_subset, ratings_subset = select_rated_subset(grids, ratings[:,i]) # gets subset of the dataset rated by advisor i
    grids_subset = grids_subset.reshape(grids_subset.shape[0], 49)

    grids_train, grids_test, ratings_train, ratings_test = train_test_split(grids_subset, ratings_subset)
    grids_train = pd.DataFrame(grids_train, columns = range(grids_subset.shape[1]), dtype = "object") # specify dtype of object to ensure categorical handling of data
    grids_test = pd.DataFrame(grids_test, columns = range(grids_subset.shape[1]), dtype = "object")
    preds_train = pd.DataFrame(ratings_train, columns = ["ratings"])
    all_train = pd.concat([grids_train, preds_train], axis=1)

    predictor = TabularPredictor(label="ratings").fit(all_train, hyperparameters = {'NN_TORCH':{'num_epochs': 100, 'weight_decay': 1e-4}, 'GBM':{'extra_trees': True, 'ag_args': {'name_suffix': 'L2', 'quantile_alpha': 0.75}}, 'RF':{}, 'XT':{}, 'CAT':{}})

    preds_test = predictor.predict(grids_test)
    preds_train = predictor.predict(grids_train)
    plot_and_r2(preds_train, preds_test, ratings_train, ratings_test, i)
    predictions = get_predictions(grids, ratings[:,i], predictor)
    return predictions, predictor

def get_pairwise_dist(mask_A, mask_B, freq = []):
    """ Returns the pairwise distance between two vectors. The elements of the input vectors are the cartesian coordinates."""
    if len(freq) == 0:
        freq = [0]
        for k1 in range(6):
            for k2 in range(7):
                freq.append((k1 + 1)**2 + (k2)**2)
        freq = np.unique(np.array(freq))
    
    distance_array = np.zeros(mask_A.shape[0] * mask_B.shape[0])
    n1 = np.shape(mask_B)[0]
    for i1 in range(np.shape(mask_A)[0]):
        dist = np.linalg.norm(mask_B - mask_A[i1] , axis = 1, ord = 2)
        distance_array[i1 * n1 : (i1 + 1) * n1] = dist
            
    sq_distance_array = np.sort(np.rint(distance_array**2))
    sq_distance_array = np.concatenate((sq_distance_array, freq))
    unique, unique_count = np.unique(sq_distance_array, return_counts = True)
    unique_count = unique_count - 1
    return unique_count

def get_statistical_moment(mask_A, mask_B, freq, rescale = False):
    if len(freq) == 0:
        freq = [0]
        for k1 in range(6):
            for k2 in range(7):
                freq.append((k1 + 1)**2 + (k2)**2)
        freq = np.unique(np.array(freq))
    
    distance_array = np.zeros(mask_A.shape[0] * mask_B.shape[0])
    sq_distance_array = np.sort(np.rint(distance_array**2))
    sq_distance_array = np.concatenate((sq_distance_array, freq))
    unique, unique_count = np.unique(sq_distance_array, return_counts = True)
    adjacent = np.sum(unique_count[1:3])
    n1 = np.shape(mask_B)[0]
    for i1 in range(np.shape(mask_A)[0]):
        dist = np.linalg.norm(mask_B - mask_A[i1] , axis = 1, ord = 2)
        distance_array[i1 * n1 : (i1 + 1) * n1] = dist

    if distance_array.shape[0] == 0: 
        return np.zeros(3)
    
    expected_dist = np.mean(distance_array)  
    if rescale == True:
        try: 
            expected_dist = expected_dist * ((n1 + 1) / n1)
        except ZeroDivisionError:
            expected_dist = expected_dist

    variance_dist = np.var((distance_array - expected_dist))   

    return [expected_dist, variance_dist, adjacent]

def get_features_moments(grids_subset, freq = []):
    num_layers = grids_subset.shape[0]
    
    if len(freq) == 0:
        freq = [0]
        for k1 in range(6):
            for k2 in range(7):
                freq.append((k1 + 1)**2 + (k2)**2)
        freq = np.unique(np.array(freq))

    freq_length = 3
    feat_out = np.zeros((num_layers, freq_length * 15 + 5))
    
    for i1 in range(num_layers):
        mask_0 = np.argwhere(grids_subset[i1] == 0)
        mask_1 = np.argwhere(grids_subset[i1] == 1)
        mask_2 = np.argwhere(grids_subset[i1] == 2)
        mask_3 = np.argwhere(grids_subset[i1] == 3)
        mask_4 = np.argwhere(grids_subset[i1] == 4)
        
        feat_out[i1, 0 * freq_length : 1 * freq_length ] = get_statistical_moment(mask_0, mask_0, freq) 
        feat_out[i1, 1 * freq_length : 2 * freq_length ] = get_statistical_moment(mask_0, mask_1, freq) 
        feat_out[i1, 2 * freq_length : 3 * freq_length ] = get_statistical_moment(mask_0, mask_2, freq) 
        feat_out[i1, 3 * freq_length : 4 * freq_length ] = get_statistical_moment(mask_0, mask_3, freq) 
        feat_out[i1, 4 * freq_length : 5 * freq_length ] = get_statistical_moment(mask_0, mask_4, freq) 
        feat_out[i1, 5 * freq_length : 6 * freq_length ] = get_statistical_moment(mask_1, mask_1, freq) 
        feat_out[i1, 6 * freq_length : 7 * freq_length ] = get_statistical_moment(mask_1, mask_2, freq) 
        feat_out[i1, 7 * freq_length : 8 * freq_length ] = get_statistical_moment(mask_1, mask_3, freq) 
        feat_out[i1, 8 * freq_length : 9 * freq_length ] = get_statistical_moment(mask_1, mask_4, freq) 
        feat_out[i1, 9 * freq_length : 10 * freq_length ] = get_statistical_moment(mask_2, mask_2, freq) 
        feat_out[i1, 10 * freq_length : 11 * freq_length ] = get_statistical_moment(mask_2, mask_3, freq) 
        feat_out[i1, 11 * freq_length : 12 * freq_length ] = get_statistical_moment(mask_2, mask_4, freq) 
        feat_out[i1, 12 * freq_length : 13 * freq_length ] = get_statistical_moment(mask_3, mask_3, freq) 
        feat_out[i1, 13 * freq_length : 14 * freq_length ] = get_statistical_moment(mask_3, mask_4, freq) 
        feat_out[i1, 14 * freq_length : 15 * freq_length ] = get_statistical_moment(mask_4, mask_4, freq) 
        
        feat_out[i1, 15 * freq_length : 15 * freq_length + 5] = [mask_0.shape[0], mask_1.shape[0], mask_2.shape[0], mask_3.shape[0], mask_4.shape[0]]
    return feat_out

def get_features_sqdist(grids_subset, freq = []):
    num_layers = grids_subset.shape[0]

    if len(freq) == 0:
        freq = [0]
        for k1 in range(6):
            for k2 in range(7):
                freq.append((k1 + 1)**2 + (k2)**2)
        freq = np.unique(np.array(freq))
    
    feat_out = np.zeros((num_layers, np.shape(freq)[0] * 15))
    freq_length = np.shape(freq)[0]
    
    for i1 in range(num_layers):
        mask_0 = np.argwhere(grids_subset[i1] == 0)
        mask_1 = np.argwhere(grids_subset[i1] == 1)
        mask_2 = np.argwhere(grids_subset[i1] == 2)
        mask_3 = np.argwhere(grids_subset[i1] == 3)
        mask_4 = np.argwhere(grids_subset[i1] == 4)
        
        feat_out[i1, 0 * freq_length : 1 * freq_length ] = get_pairwise_dist(mask_0, mask_0, freq)
        feat_out[i1, 1 * freq_length : 2 * freq_length ] = get_pairwise_dist(mask_0, mask_1, freq)
        feat_out[i1, 2 * freq_length : 3 * freq_length ] = get_pairwise_dist(mask_0, mask_2, freq)
        feat_out[i1, 3 * freq_length : 4 * freq_length ] = get_pairwise_dist(mask_0, mask_3, freq)
        feat_out[i1, 4 * freq_length : 5 * freq_length ] = get_pairwise_dist(mask_0, mask_4, freq)
        feat_out[i1, 5 * freq_length : 6 * freq_length ] = get_pairwise_dist(mask_1, mask_1, freq) 
        feat_out[i1, 6 * freq_length : 7 * freq_length ] = get_pairwise_dist(mask_1, mask_2, freq)
        feat_out[i1, 7 * freq_length : 8 * freq_length ] = get_pairwise_dist(mask_1, mask_3, freq)
        feat_out[i1, 8 * freq_length : 9 * freq_length ] = get_pairwise_dist(mask_1, mask_4, freq)
        feat_out[i1, 9 * freq_length : 10 * freq_length ] = get_pairwise_dist(mask_2, mask_2, freq) 
        feat_out[i1, 10 * freq_length : 11 * freq_length ] = get_pairwise_dist(mask_2, mask_3, freq)
        feat_out[i1, 11 * freq_length : 12 * freq_length ] = get_pairwise_dist(mask_2, mask_4, freq)
        feat_out[i1, 12 * freq_length : 13 * freq_length ] = get_pairwise_dist(mask_3, mask_3, freq) 
        feat_out[i1, 13 * freq_length : 14 * freq_length ] = get_pairwise_dist(mask_3, mask_4, freq)
        feat_out[i1, 14 * freq_length : 15 * freq_length ] = get_pairwise_dist(mask_4, mask_4, freq)
 
    return feat_out

def fit_plot_predict_zyzh(grids, ratings, i):
    """ 
    Function to implement autoML to correlate test data and labels, for a given advisor i
    Returns the Prediction (model y-values) and the Predictor (Trained Model)
    """
    grids_subset, ratings_subset = select_rated_subset(grids, ratings[:,i]) # gets subset of the dataset rated by advisor i

    ## Feature transformation 
    freq = [0]
    for k1 in range(6):
        for k2 in range(7):
            freq.append((k1 + 1)**2 + (k2)**2)
    freq = np.unique(np.array(freq))

    grids_subset_train, grids_subset_test, ratings_train, ratings_test = train_test_split(grids_subset, ratings_subset)


    feats_train = get_features_sqdist(grids_subset_train, freq)
    feats_test = get_features_sqdist(grids_subset_test, freq)

    
    feats_train = pd.DataFrame(feats_train, columns = range(feats_train.shape[1]), dtype = "object") # specify dtype of object to ensure categorical handling of data
    feats_test = pd.DataFrame(feats_test, columns = range(feats_test.shape[1]), dtype = "object")
    preds_train = pd.DataFrame(ratings_train, columns = ["ratings"])
    all_train = pd.concat([feats_train, preds_train], axis=1)

    predictor = TabularPredictor(label="ratings").fit(all_train, hyperparameters = {'NN_TORCH':{'num_epochs': 100, 'weight_decay': 1e-4}, 'XGB':{'reg_lambda': 0.1}, 'RF':{}, 'XT':{}, 'CAT':{'l2_leaf_reg': 3.0}})

    preds_test = predictor.predict(feats_test)
    preds_train = predictor.predict(feats_train)

    np.savetxt('prediction.npy', preds_test, delimiter=',')
    np.savetxt('ground_truth.npy', ratings_test, delimiter=',')
    np.savez('grids.npz', grids_subset_test, delimiter=',')

    plot_and_r2(preds_train, preds_test, ratings_train, ratings_test, i)

    feats_all = np.loadtxt('feats.csv', delimiter = ',')
    predictions = get_predictions(feats_all, ratings[:,i], predictor)
    return predictions, predictor

## Implementing ML model 
grids = load_grids()                                            # Helper function we have provided to load the grids from the dataset
ratings = np.load("datasets/scores.npy")                        # Load advisor scores
score_order = ["Wellness", "Tax", "Transportation", "Business"] # This is the order of the scores in the dataset
ratings_df = pd.DataFrame(ratings, columns = score_order)       # Create a dataframe

all_predictions = []
all_predictors = []

for i in range(2,3):
    predictions, predictor = fit_plot_predict_zyzh(grids, ratings, i)
    all_predictions.append(predictions)
    all_predictors.append(predictor)

