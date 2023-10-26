from utils_public import *
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Define the grid dimensions
grid_size = 7
total_nodes = grid_size * grid_size

# Initialize an empty adjacency matrix
adjacency_matrix = np.zeros((total_nodes, total_nodes), dtype=int)

# Function to check if a node is within the grid
def is_valid_node(x, y):
    return 0 <= x < grid_size and 0 <= y < grid_size

# Iterate over all nodes in the grid
for node in range(total_nodes):
    row, col = node // grid_size, node % grid_size
    
    # Define the possible neighbors (including diagonals)
    neighbors = [
        (row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1),
        (row - 1, col - 1), (row - 1, col + 1), (row + 1, col - 1), (row + 1, col + 1)
    ]
    
    # Iterate over neighbors and set connections in the adjacency matrix
    for neighbor in neighbors:
        n_row, n_col = neighbor
        if is_valid_node(n_row, n_col):
            neighbor_node = n_row * grid_size + n_col
            adjacency_matrix[node, neighbor_node] = 1

# np.savetxt(fname = 'adjacency.csv', X = adjacency_matrix, delimiter = ',')

## Implement GNN
class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, adj_matrix):
        # Apply linear transformation
        x_transformed = self.linear(x)
        x_transformed = x_transformed.unsqueeze(0)

        # Expand the adjacency matrix dimensions for batch multiplication
        adj_matrix_expanded = adj_matrix.unsqueeze(0).expand(x_transformed.size(0), -1, -1)

        # Use batch matrix multiplication
        x = torch.bmm(adj_matrix_expanded, x_transformed)
        
        return x


class GraphNeuralNetwork(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(GraphNeuralNetwork, self).__init__()
        
        # Define your graph convolutional layers
        self.gc1 = GraphConvLayer(input_channels, hidden_channels)
        self.gc2 = GraphConvLayer(hidden_channels, hidden_channels)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_channels, output_channels)

    def forward(self, x, adj_matrix):
        # Apply graph convolutional layers with activation functions
        x = F.relu(self.gc1(x, adj_matrix))
        x = F.relu(self.gc2(x, adj_matrix))
        
        # Apply output layer
        x = self.output_layer(x)
        
        return x

# Instantiate your GNN model
GNNmodel = GraphNeuralNetwork(input_channels = total_nodes, hidden_channels = 64, output_channels = 1)
num_adv = 4

grids = load_grids()                                            # Helper function we have provided to load the grids from the dataset
ratings = np.load("datasets/scores.npy")                        # Load advisor scores
score_order = ["Wellness", "Tax", "Transportation", "Business"] # This is the order of the scores in the dataset
ratings_df = pd.DataFrame(ratings, columns = score_order)       # Create a dataframe

# Forward pass
for i1 in range(1):
    grids_subset, ratings_subset = select_rated_subset(grids, ratings[:,i1]) # gets subset of the dataset rated by advisor i
    grids_train, grids_test, ratings_train, ratings_test = train_test_split(grids_subset, ratings_subset, test_size = 0.2)
    grids_train_tensor = torch.tensor(grids_train.reshape(len(grids_train), -1), dtype = torch.float64)
    ratings_train_tensor = torch.tensor(ratings_train, dtype = torch.float64).unsqueeze(-1) # Add extra dimension
    adjacency_tensor = torch.tensor(adjacency_matrix, dtype = torch.float64)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(GNNmodel.parameters(), lr = 0.01)
    num_epochs = 100
    batch_size = 32
    train_dataset = TensorDataset(grids_train_tensor, ratings_train_tensor)
    print(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    
    # Training loop
    for epoch in range(num_epochs):
        for batch_grids, batch_ratings in train_loader:
            # Forward pass
            outputs = GNNmodel(batch_grids, adjacency_tensor)

            # Compute loss
            loss = criterion(outputs, batch_ratings)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:   
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


    grids_test_tensor = torch.tensor(grids_test.reshape(len(grids_test), -1), dtype=torch.float32)
    ratings_test_tensor = torch.tensor(ratings_test, dtype=torch.float32).unsqueeze(-1)

    GNNmodel.eval()

    with torch.no_grad():  # Disabling gradient computation, saves memory and speeds up computation
        test_outputs = GNNmodel(grids_test_tensor, adjacency_tensor)

    # Evaluate the predictions
    test_loss = criterion(test_outputs, ratings_test_tensor)
    print(f"Test Loss: {test_loss.item():.4f}")
