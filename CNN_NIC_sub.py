from utils_public import *
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, concatenate
from sklearn.metrics import r2_score
from multiprocessing import Pool, cpu_count
from functools import partial
from scipy.spatial import distance
import itertools
import math


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

def compute_features(grid, advisor):

    features = []
    grid = grid.astype(int)
    # Number of each type
    counts = np.bincount(grid.flatten(), minlength=5)
    features.extend(counts)
    # number of times a cell of class i is adjacent to a cell of class j.
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

    for key, values in distances.items():
        if values:  # Check if the list is not empty
            try:
                mean_val = np.mean(values)
                var_val = np.var(values)
            except:
                mean_val = 0  
                var_val = 0 
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
        for cls in [0, 1, 2, 4]:
                   
            # obtains the minimum and max distance between residential areas and three others
            # max_min = compute_max_min_distance(grid, class_type_1 = 0, class_type_2 = cls)
            # features.extend(max_min)
            # obtains distance to nearest residential
            distance_matrix = compute_distance_to_class(grid, target_class = cls)
            features.extend(distance_matrix)

        # could possibly add proximity of houses at the centre
    num_classes = 5
    centroid_list = np.zeros(num_classes)   
    largest_sizes, centroid_dict, cluster_points = find_largest_clusters(grid, num_classes)
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

        # largest_sizes, centroid_dict, cluster_points = find_largest_clusters(grid, num_classes)
        # min_distances, max_distances = pairwise_distances_between_lists(cluster_points)
        # features.extend(min_distances)
        # features.extend(max_distances)

        # # Calculate centroid distances
        # centroid_distances = []
        # for i in range(num_classes):
        #     for j in range(i + 1, num_classes):
        #         if np.isnan(centroid_list[i]).any() or np.isnan(centroid_list[j]).any():
        #             centroid_distances.append(0)
        #         else:
        #             centroid_distances.append(distance.euclidean(centroid_list[i], centroid_list[j]))
        #         
        # # Append centroid distances to features
        # features.extend(centroid_distances)
    return features


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

def one_hot_encode(grid):
    # The shape of grids_oh will be (7, 7, 5) after one-hot encoding
    grids_oh = (np.arange(5) == grid[..., None]).astype(int)
    return grids_oh

def plot_and_r2(preds_train, preds_test, ratings_train, ratings_test, i):
    plt.scatter(ratings_train, preds_train, label='Train Set Preds', s=3, c = "#F08E18") #Train set in orange
    plt.scatter(ratings_test, preds_test, label='Test Set Preds', s=5, c = "#DC267F") #Test set in magenta
    plt.plot([0,1], [0,1], label="target", linewidth=3, c="k") # Target line in Black
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.title(f"Advisor {i} Predictions")
    plt.legend()
    plt.show()
    print(f"Train Set R2 score: {r2_score(ratings_train, preds_train)}") #Calculate R2 score
    print(f"Test Set R2 score: {r2_score(ratings_test, preds_test)}")

from multiprocessing import Pool, cpu_count
from functools import partial

def parallel_compute(grids, advisor):
    with Pool(cpu_count()) as p:
        # Partially apply the advisor to the compute_features function
        func = partial(compute_features, advisor=advisor)
        # Map the function over grids
        return np.array(p.starmap(func, [(grid,) for grid in grids]))


# Optimal hyperparameters: {'batch_size': 64, 'conv_layer_size': 92, 'dense_layer_size': 84, 'epochs': 29, 'learning_rate': 0.003837244776335524, 'num_conv_layers': 2, 'num_dense_layers': 4}
# Optimal hyperparameters: {'batch_size': 64, 'conv_layer_size': 171, 'dense_layer_size': 84, 'epochs': 29, 'learning_rate': 0.003837244776335524, 'num_conv_layers': 2, 'num_dense_layers': 4}


def create_combined_model(num_conv_layers=2, conv_layer_size=171, num_dense_layers=4, dense_layer_size=84, learning_rate=0.003837244776335524, advisor = 0):

    # Convolutional Branch
    input_grid = Input(shape=(7, 7, 5))
    x = input_grid
    
    # Add convolutional layers dynamically
    for _ in range(num_conv_layers):
        x = Conv2D(conv_layer_size, (3, 3), activation='relu', padding='same')(x)
        
    x = Flatten()(x)
    x = Dense(dense_layer_size, activation='relu')(x)
    conv_branch = Dense(dense_layer_size, activation='relu')(x)
    
    # Dense Branch
    if advisor == 0:
        input_features = Input(shape=(75,))
    if advisor == 1:
        input_features = Input(shape=(25,))
    if advisor == 2:
        input_features = Input(shape=(221,))
    if advisor == 3:
        input_features = Input(shape=(25,))
    y = input_features
    
    # Add dense layers dynamically
    for _ in range(num_dense_layers):
        y = Dense(dense_layer_size, activation='relu')(y)
        
    dense_branch = Dense(dense_layer_size, activation='relu')(y)
    
    # Combining the two branches
    combined = concatenate([conv_branch, dense_branch])
    combined = Dense(dense_layer_size, activation='relu')(combined)
    combined = Dense(int(dense_layer_size/2), activation='relu')(combined)
    output = Dense(1)(combined)

    # Use the learning rate in the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Create the model
    model = Model(inputs=[input_grid, input_features], outputs=output)
    
    # Compile the model with the custom optimizer
    model.compile(optimizer=optimizer, loss='mse')
   
    
    return model

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

