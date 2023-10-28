
from utils_public import *
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, concatenate
from sklearn.metrics import r2_score
from multiprocessing import Pool, cpu_count
from functools import partial
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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

    
    return features

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

def parallel_compute(grids, advisor):
    with Pool(cpu_count()) as p:
        # Partially apply the advisor to the compute_features function
        func = partial(compute_features, advisor=advisor)
        # Map the function over grids
        return np.array(p.starmap(func, [(grid,) for grid in grids]))
    

# Optimal hyperparameters: {'batch_size': 64, 'conv_layer_size': 92, 'dense_layer_size': 84, 'epochs': 29, 'learning_rate': 0.003837244776335524, 'num_conv_layers': 2, 'num_dense_layers': 4}
# Optimal hyperparameters: {'batch_size': 64, 'conv_layer_size': 171, 'dense_layer_size': 84, 'epochs': 29, 'learning_rate': 0.003837244776335524, 'num_conv_layers': 2, 'num_dense_layers': 4}


def create_combined_model(num_conv_layers=2, conv_layer_size=171, num_dense_layers=4, dense_layer_size=84, learning_rate=0.003837244776335524):
    
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
    input_features = Input(shape=(216,))
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
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Create the model
    model = Model(inputs=[input_grid, input_features], outputs=output)
    
    # Compile the model with the custom optimizer
    model.compile(optimizer=optimizer, loss='mse')
   
    
    return model

grids = load_grids() #Helper function we have provided to load the grids from the dataset
ratings = np.load("datasets/scores.npy") #Load advisor scores
score_order = ["Wellness", "Tax", "Transportation", "Business"] #This is the order of the scores in the dataset
ratings_df = pd.DataFrame(ratings, columns = score_order) #Create a dataframe

advisor_val = 2
grids_subset, ratings_subset = select_rated_subset(grids, ratings[:,advisor_val]) #gets subset of the dataset rated by advisor 2
grids_encoded = np.array([one_hot_encode(grid) for grid in grids_subset])

# First split: 80% for training, 20% for temp (to be divided into test and validation)
grids_train, grids_test, ratings_train, ratings_test = train_test_split(grids_encoded, ratings_subset, test_size=0.2, random_state=20)

features_subset = parallel_compute(grids_subset, advisor_val)
features_subset[np.isnan(features_subset)] = 0
features_train, features_test, ratings_train, ratings_test = train_test_split(features_subset, ratings_subset, test_size=0.2, random_state=20)

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
