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
import itertools
import math

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

def get_feats_all(grids, freq = []):
    if len(freq) == 0:
        freq = [0]
        for k1 in range(6):
            for k2 in range(7):
                freq.append((k1 + 1)**2 + (k2)**2)
        freq = np.unique(np.array(freq))
    feats_all = get_features_sqdist(grids, freq)
    np.savetxt(X = feats_all, fname = 'feats.csv', delimiter = ',')

def compute_features(grids, advisor):
    num_layers = grids.shape[0]

    feature_all = np.zeros((num_layers,290))
    for i1 in range(num_layers):
        grid = grids[i1,:,:]
        features = []
        grid = grid.astype(int)

        # Number of each type
        counts = np.bincount(grid.flatten(), minlength=5)
        features.extend(counts)

        inter_adjacency = np.zeros((5, 5), dtype=int)

        for i in range(7):
            for j in range(7):
                current_val = grid[i, j]
                for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                    if 0 <= i+dx < 7 and 0 <= j+dy < 7:
                        neighbor_val = grid[i+dx, j+dy]
                        # Only update the upper diagonal (including main diagonal)
                        if current_val <= neighbor_val:
                            inter_adjacency[current_val, neighbor_val] += 1

        features.extend(inter_adjacency[np.triu_indices(5, k=0)])

        # Create a dictionary to store distances for each unique class pair
        distances = {(i, j): [] for i in range(5) for j in range(i, 5)}

        # Compute distances for each pair of cells
        for i in range(7):
            for j in range(7):
                for m in range(7):
                    for n in range(7):
                        d = np.sqrt((i - m) ** 2 + (j - n) ** 2)
                        class_pair = tuple(sorted([grid[i, j], grid[m, n]]))
                        distances[class_pair].append(d)

        # Compute statistics for each class pair and store in a flattened list
        flattened_stats = []
        # for key, values in distances.items():
        #     mean_val = np.mean(values)
        #     var_val = np.var(values)
        #     # skew_val = skew(values)
        #     # kurt_val = kurtosis(values)
        #     # flattened_stats.extend([mean_val, var_val, skew_val, kurt_val])        
        #     flattened_stats.extend([mean_val, var_val])
        # features.extend(flattened_stats)

        for key, values in distances.items():
            if values:  # Check if the list is not empty
                try:
                    mean_val = np.mean(values)
                    var_val = np.var(values)
                except:
                    mean_val = 0  # or any default value
                    var_val = 0  # or any default value
                flattened_stats.extend([mean_val, var_val])
            else:  # Handle the case where the list is empty
                flattened_stats.extend([0, 0])  # or any default value

        if advisor == 0:
            " Wellness advisor is focused on the health and wellbeing (both physical and mental) of citizens"
            " They are very invested in the quality and accessibility of city's green spaces."
            # Distance of park zones from each grid element
            distance_matrix = compute_distance_to_class(grid, target_class = 3)
            features.extend(distance_matrix)
            # Number of connected parks
            connections = count_connected_for_class(grid, target_class= 3)
            features.extend([connections])

        if advisor == 2:
            " Transportation advisor places an emphasis on accessibility and emissions. "
            " They are focused on mimizing the distance over which the workforce needs to commute."

            
            num_classes = 5
            centroid_list = np.zeros(num_classes)
            for cls in [0, 1, 2, 3, 4]:

                # obtains the minimum and max distance between residential areas and three others
                # max_min = compute_max_min_distance(grid, class_type_1 = 0, class_type_2 = cls)
                # features.extend(max_min)
                # obtains distance to nearest residential
                distance_matrix = compute_distance_to_class(grid, target_class = cls)
                features.extend(distance_matrix)
                # calculate clusters above a min size
                # cluster_count = count_clusters_above_size(grid, cls, min_size = 2)
                # features.append(cluster_count)
                # could possibly add proximity of houses at the centre
                # find whether the orientation matters 

                ## largest_cluster_size = find_largest_cluster(grid, cls)
                ## features.append(largest_cluster_size)
            # Find the largest clusters

            largest_sizes, centroid_dict, cluster_points = find_largest_clusters(grid, num_classes)
            min_distances, max_distances = pairwise_distances_between_lists(cluster_points)
            features.extend(min_distances)
            # features.extend(max_distances)

            
            # Create a list to store centroid tuples
            centroid_list = [centroid_dict[cls] for cls in range(num_classes)]


            # Perform element-wise division and set to NaN where division by zero occurs
            for i in range(len(centroid_list)):
                for j in range(len(centroid_list)):
                    if largest_sizes[i] != 0:
                        centroid_list[i] = tuple(coord / largest_sizes[i] for coord in centroid_list[i])
                    else:
                        centroid_list[i] = (np.nan, np.nan)

     
            # Append the largest cluster sizes
            features.extend(largest_sizes)
            
            # Calculate centroid distances
            centroid_distances = []
            for i in range(num_classes):
                for j in range(i + 1, num_classes):
                    if np.isnan(centroid_list[i]).any() or np.isnan(centroid_list[j]).any():
                        centroid_distances.append(0)
                    else:
                        centroid_distances.append(distance.euclidean(centroid_list[i], centroid_list[j]))
            
            # Append centroid distances to features
            features.extend(centroid_distances)


               
        features = np.array(features)
        feature_all[i1,:] = features
    return feature_all

def count_connected_for_class(grid, target_class):
    height, width = grid.shape
    count = 0

    for i in range(height):
        for j in range(width):
            if grid[i][j] == target_class:
                # Check bottom neighbor
                if i + 1 < height and grid[i+1][j] == target_class:
                    count += 1

                # Check right neighbor
                if j + 1 < width and grid[i][j+1] == target_class:
                    count += 1

    return count

def compute_distance_to_class(grid, target_class):
    # Find the positions of the target class
    positions_target_class = np.argwhere(grid == target_class)
    
    # Initialize a distance matrix with large values
    distance_matrix = np.full(grid.shape, np.inf)
    
    for i in range(7):
        for j in range(7):
            for pos in positions_target_class:
                distance = np.linalg.norm(np.array([i, j]) - pos)
                distance_matrix[i, j] = min(distance_matrix[i, j], distance)
    
    return distance_matrix.flatten()

def compute_max_min_distance(grid, class_type_1, class_type_2):
    # Find the positions of the two class types
    positions_type_1 = np.argwhere(grid == class_type_1)
    positions_type_2 = np.argwhere(grid == class_type_2)

    # If either class type is not found in the grid, return None for min and max distances
    if len(positions_type_1) == 0 or len(positions_type_2) == 0:
        return None, None

    # Calculate all pairwise distances between the two sets of positions
    distances = []
    for pos1 in positions_type_1:
        for pos2 in positions_type_2:
            distance = np.linalg.norm(pos1 - pos2)
            distances.append(distance)

    # Return the minimum and maximum of the calculated distances
    return [float(min(distances)), float(max(distances))]

def fit_plot_predict_zyzh(grids, ratings, i):
    """ 
    Function to implement autoML to correlate test data and labels, for a given advisor i
    Returns the Prediction (model y-values) and the Predictor (Trained Model)
    """    
    grids_subset, ratings_subset = select_rated_subset(grids, ratings[:,i]) # gets subset of the dataset rated by advisor i
    grids_subset_train, grids_subset_test, ratings_train, ratings_test = train_test_split(grids_subset, ratings_subset)

    feats_train = compute_features(grids_subset_train, i)
    feats_test = compute_features(grids_subset_test, i)
    ## Feature transformation 
    freq = [0]
    for k1 in range(6):
        for k2 in range(7):
            freq.append((k1 + 1)**2 + (k2)**2)
    freq = np.unique(np.array(freq))

    # get_feats_all(grids, freq)            # Code to generate feats_all for old ZY-ZH features
    # feats = compute_features(grids_subset, i)
    # feats_train, feats_test, ratings_train, ratings_test = train_test_split(feats, ratings_subset)
    feats_train = pd.DataFrame(feats_train, columns = range(feats_train.shape[1]), dtype = "object") # specify dtype of object to ensure categorical handling of data
    feats_test = pd.DataFrame(feats_test, columns = range(feats_test.shape[1]), dtype = "object")
    preds_train = pd.DataFrame(ratings_train, columns = ["ratings"])
    all_train = pd.concat([feats_train, preds_train], axis=1)

    # predictor = TabularPredictor(label="ratings", sample_weight = 'auto_weight', eval_metric = 'r2').fit(all_train, hyperparameters = {'NN_TORCH':{'weight_decay': 1e-4}, 'XGB':{'reg_lambda': 0.1}, 'RF':{}, 'XT':{}, 'CAT':{'l2_leaf_reg': 3.0}})
    predictor = TabularPredictor(label="ratings", sample_weight = 'auto_weight', eval_metric = 'r2').fit(all_train, hyperparameters = {'NN_TORCH':{'weight_decay': 1e-4}, 'XGB':{'reg_lambda': 0.1}, 'RF':{}, 'XT':{}})
    


    preds_test = predictor.predict(feats_test)
    preds_train = predictor.predict(feats_train)
    np.savetxt('prediction_nic.csv', preds_test, delimiter=',')
    np.savetxt('ground_truth_nic.csv', ratings_test, delimiter=',')
    np.savez('grids.npz', grids_subset_test, delimiter=',')

 
    plot_and_r2(preds_train, preds_test, ratings_train, ratings_test, i)
    # feats_all = compute_features(grids, i)
    # feats_all = np.loadtxt('feats.csv', delimiter = ',')      # For old ZY-ZH features
    # predictions = get_predictions(feats_all, ratings[:,i], predictor)
    return predictions, predictor

def find_largest_clusters(grid, num_classes):
    largest_cluster_sizes = np.zeros(num_classes, dtype=int)
    centroids = {}
    largest_cluster_points = []
       
    for target_value in range(num_classes):
        visited = np.zeros_like(grid)
        largest_cluster_size = 0
        largest_cluster_centroid = (0, 0)
        largest_cluster_point = []
        
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                if grid[row][col] == target_value and not visited[row][col]:
                    cluster_size, cluster_centroid, cluster_point= dfs(grid, row, col, target_value, visited)
                    if cluster_size > largest_cluster_size:
                        largest_cluster_size = cluster_size
                        largest_cluster_centroid = cluster_centroid
                        largest_cluster_point = cluster_point
                        

        largest_cluster_sizes[target_value] = largest_cluster_size
        centroids[target_value] = largest_cluster_centroid
        largest_cluster_points.append([point for point in largest_cluster_point if point])
        

    return largest_cluster_sizes, centroids, largest_cluster_points

def dfs(grid, row, col, target, visited):
    if (
        0 <= row < len(grid)
        and 0 <= col < len(grid[0])
        and grid[row][col] == target
        and not visited[row][col]
    ):
        visited[row][col] = 1
        
        
        size = 1
        centroid_x, centroid_y = row, col
        points = [(row,col)]
        # include diagonal as cluster
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                size_delta, centroid_delta, point_delta = dfs(grid, row + dr, col + dc, target, visited)
                size += size_delta
                centroid_x += centroid_delta[0]
                centroid_y += centroid_delta[1]
                points.extend(point_delta)
                
        # Explore only adjacent cells (up, down, left, and right)
        # directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        # for dr, dc in directions:
        #     size_delta, centroid_delta = dfs(grid, row + dr, col + dc, target, visited)
        #     size += size_delta
        #     centroid_x += centroid_delta[0]
        #     centroid_y += centroid_delta[1]




        return size, (centroid_x, centroid_y), points

    return 0, (0, 0), ()

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def pairwise_distances_between_lists(lists_of_points):
    num_lists = len(lists_of_points)
    min_distances = [[float('inf')] * num_lists for _ in range(num_lists)]
    max_distances = [[0] * num_lists for _ in range(num_lists)]

    for i, j in itertools.combinations(range(num_lists), 2):
        for point1 in lists_of_points[i]:
            for point2 in lists_of_points[j]:
                distance = euclidean_distance(point1, point2)
                min_distances[i][j] = min(min_distances[i][j], distance)
                min_distances[j][i] = min_distances[i][j]  # Symmetric
                max_distances[i][j] = max(max_distances[i][j], distance)
                max_distances[j][i] = max_distances[i][j]  # Symmetric

    # Extract the upper triangular parts of the matrices
    upper_triangular_min_distances = [min_distances[i][j] for i, j in itertools.combinations(range(num_lists), 2)]
    upper_triangular_max_distances = [max_distances[i][j] for i, j in itertools.combinations(range(num_lists), 2)]

    # Return 1D arrays
    return upper_triangular_min_distances, upper_triangular_max_distances


def count_clusters_above_size(grid, target_value, min_size):
    def dfs2(row, col):
        if (
            row < 0
            or col < 0
            or row >= len(grid)
            or col >= len(grid[0])
            or grid[row][col] != target_value
            or visited[row][col]  # Check if cell is already visited
        ):
            return 0
        visited[row][col] = True  # Mark the cell as visited
        count = 1
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            count += dfs2(row + dr, col + dc)
        return count

    cluster_count = 0
    visited = [[False] * len(grid[0]) for _ in range(len(grid))]  # Initialize visited matrix
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] == target_value and not visited[row][col]:
                cluster_size = dfs2(row, col)
                if cluster_size >= min_size:
                    cluster_count += 1

    return cluster_count
    