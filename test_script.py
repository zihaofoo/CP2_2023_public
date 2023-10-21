import numpy as np
from utils_public import *
from scipy.spatial.distance import pdist, squareform
import pdb

def get_pairwise_dist(mask_A, mask_B, freq = []):
    """ Returns the pairwise distance between two vectors. The elements of the input vectors are the cartesian coordinates."""
    if len(freq) == 0:
        freq = []
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


def get_features(grids_subset, freq = []):
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
        
        
        feat_out[i1, 0 * freq_length : 1 * freq_length ] = get_pairwise_dist(mask_0, mask_0, freq) / 2
        feat_out[i1, 1 * freq_length : 2 * freq_length ] = get_pairwise_dist(mask_0, mask_1, freq)
        feat_out[i1, 2 * freq_length : 3 * freq_length ] = get_pairwise_dist(mask_0, mask_2, freq)
        feat_out[i1, 3 * freq_length : 4 * freq_length ] = get_pairwise_dist(mask_0, mask_3, freq)
        feat_out[i1, 4 * freq_length : 5 * freq_length ] = get_pairwise_dist(mask_0, mask_4, freq)
        feat_out[i1, 5 * freq_length : 6 * freq_length ] = get_pairwise_dist(mask_1, mask_1, freq) / 2
        feat_out[i1, 6 * freq_length : 7 * freq_length ] = get_pairwise_dist(mask_1, mask_2, freq)
        feat_out[i1, 7 * freq_length : 8 * freq_length ] = get_pairwise_dist(mask_1, mask_3, freq)
        feat_out[i1, 8 * freq_length : 9 * freq_length ] = get_pairwise_dist(mask_1, mask_4, freq)
        feat_out[i1, 9 * freq_length : 10 * freq_length ] = get_pairwise_dist(mask_2, mask_2, freq) / 2
        feat_out[i1, 10 * freq_length : 11 * freq_length ] = get_pairwise_dist(mask_2, mask_3, freq)
        feat_out[i1, 11 * freq_length : 12 * freq_length ] = get_pairwise_dist(mask_2, mask_4, freq)
        feat_out[i1, 12 * freq_length : 13 * freq_length ] = get_pairwise_dist(mask_3, mask_3, freq) / 2
        feat_out[i1, 13 * freq_length : 14 * freq_length ] = get_pairwise_dist(mask_3, mask_4, freq)
        feat_out[i1, 14 * freq_length : 15 * freq_length ] = get_pairwise_dist(mask_4, mask_4, freq) / 2

    return feat_out

def fit_plot_predict_zyzh(grids, ratings, i):
    """ Function to implement autoML to correlate test data and labels, for a given advisor i"""
    grids_subset, ratings_subset = select_rated_subset(grids, ratings[:,i]) # gets subset of the dataset rated by advisor i

    ## Feature transformation 
    freq = [0]
    for k1 in range(6):
        for k2 in range(7):
            freq.append((k1 + 1)**2 + (k2)**2)
    freq = np.unique(np.array(freq))
    feats = get_features(grids_subset, freq)

    feats_train, feats_test, ratings_train, ratings_test = train_test_split(feats, ratings_subset)
    feats_train = pd.DataFrame(feats_train, columns = range(feats.shape[1]), dtype = "object") # specify dtype of object to ensure categorical handling of data
    feats_test = pd.DataFrame(feats_test, columns = range(feats.shape[1]), dtype = "object")
    preds_train = pd.DataFrame(ratings_train, columns = ["ratings"])
    all_train = pd.concat([feats_train, preds_train], axis=1)

    predictor = TabularPredictor(label="ratings").fit(all_train, hyperparameters = {'NN_TORCH':{'num_epochs': 100, 'weight_decay': 1e-4}, 'GBM':{'extra_trees': True, 'ag_args': {'name_suffix': 'L2', 'quantile_alpha': 0.75}}, 'RF':{}, 'XT':{}, 'KNN':{}})

    preds_test = predictor.predict(feats_test)
    preds_train = predictor.predict(feats_train)
    plot_and_r2(preds_train, preds_test, ratings_train, ratings_test, i)
    predictions = get_predictions(grids, ratings[:,i], predictor)
    return predictions, predictor

grids = load_grids() 
ratings = np.load("datasets/scores.npy")                        # Load advisor scores
i = 0 
grids_subset, ratings_subset = select_rated_subset(grids, ratings[:,i]) # gets subset of the dataset rated by advisor i
feat_out = get_features(grids_subset)
np.savetxt('featout.csv', feat_out, delimiter=',', fmt='%d')

print(feat_out.shape)

