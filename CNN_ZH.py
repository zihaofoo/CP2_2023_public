from utils_public import *
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import r2_score
import torch.optim as optim

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

# One-hot encoding function
def one_hot_encode(grid):
    one_hot_grid = (np.arange(5) == grid[..., None]).astype(np.float32)
    one_hot_grid = np.moveaxis(one_hot_grid, -1, 1)  # Channels first (C, H, W)
    return one_hot_grid


# CNN Model Definition
class CNN(nn.Module):
    def __init__(self, filter_size:int, kernel_size:int, dense_units:int):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(5, filter_size, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(filter_size, filter_size * 2, kernel_size, padding=kernel_size//2)
        self.fc1 = nn.Linear((filter_size * 2) * 7 * 7, dense_units)
        self.fc2 = nn.Linear(dense_units, 1)  # Change 10 to the number of classes you have

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid activation for the output layer
        return x

# Assume the cfg is a dict with the required hyperparameters
cfg = {'filter_size': 64, 'kernel_size': 3, 'dense_units': 128}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(cfg['filter_size'], cfg['kernel_size'], cfg['dense_units']).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Data Loading
grids = load_grids()                                            # Helper function we have provided to load the grids from the dataset
advisor = 2
ratings = np.load("datasets/scores.npy")                        # Load advisor scores
score_order = ["Wellness", "Tax", "Transportation", "Business"] # This is the order of the scores in the dataset

grids_subset, ratings_subset = select_rated_subset(grids, ratings[:,advisor]) # gets subset of the dataset rated by advisor i
num_layers = grids_subset.shape[0]
ratings_subset = ratings_subset[..., np.newaxis]
grids_subset = grids_subset[..., np.newaxis]
grids_subset = grids_subset.astype(int)
grids_subset = np.eye(5)[grids_subset]  # One-hot encoding
grids_subset = grids_subset.reshape(num_layers, 5, 7, 7)  # Changing to (batch, channel, height, width) format for PyTorch

# Convert to PyTorch tensors
grids_subset_tensor = torch.tensor(grids_subset, dtype=torch.float32)
ratings_subset_tensor = torch.tensor(ratings_subset, dtype=torch.float32)

# Splitting data into train and test
test_train_split = 0.8
dataset = TensorDataset(grids_subset_tensor, ratings_subset_tensor)
train_size = int(test_train_split * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)







# DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Training loop
model.train()
for epoch in range(5):  # Replace 5 with the desired number of epochs
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).unsqueeze(1)  # Ensure target is the correct shape
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)  # Use MSE loss for regression
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:  # Logging every 100 batches
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Validation loop
model.eval()
val_loss = 0
with torch.no_grad():
    for data, target in val_loader:
        data, target = data.to(device), target.to(device).unsqueeze(1)  # Ensure target is the correct shape
        output = model(data)
        val_loss += F.mse_loss(output, target, reduction='sum').item()  # sum up batch loss

val_loss /= len(val_loader.dataset)

print(f'\nValidation set: Average loss: {val_loss:.4f}\n')

