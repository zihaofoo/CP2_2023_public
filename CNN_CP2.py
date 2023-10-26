from utils_public import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import r2_score
from datetime import datetime
import GPyOpt


def predict(model, input_data):
    """
    Predict the ratings using the trained model.

    Args:
    - model (nn.Module): Trained PyTorch model.
    - input_data (Tensor): Input grids data.

    Returns:
    - Tensor: Predicted ratings.
    """
    with torch.no_grad():
        predictions = model(input_data)
    return predictions

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
    plt.show()
    print(f"Train Set R2 score: {r2_score(ratings_train, preds_train)}") #Calculate R2 score
    print(f"Test Set R2 score: {r2_score(ratings_test, preds_test)}")


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

# Model Architecture
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(5, 32, 3, 1, padding = 1)  
        self.fc1 = nn.Linear(32 * 7 * 7, 64)  
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

model = CNNModel()

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), weight_decay = 0.001)

# Training the Model
epochs = 50
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
    
    train_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")

# Evaluate on Test Data
model.eval()
test_loss = 0.0
with torch.no_grad():
    for data, target in test_loader:
        outputs = model(data)
        loss = criterion(outputs, target)
        test_loss += loss.item() * data.size(0)
test_loss /= len(test_loader.dataset)
print(f"Test Loss: {test_loss:.4f}")

preds_train = []
preds_test = []
ratings_train = []
ratings_test = []

with torch.no_grad():
    for data, target in DataLoader(train_dataset, batch_size = 32):  # Iterating over batches
        outputs = model(data)
        preds_train.append(outputs)
        ratings_train.append(target)

    for data, target in DataLoader(test_dataset, batch_size = 32):  # Iterating over batches
        outputs = model(data)
        preds_test.append(outputs)
        ratings_test.append(target)

preds_train = torch.cat(preds_train).numpy()  # concatenate batches and convert to numpy array
preds_test = torch.cat(preds_test).numpy()  # concatenate batches and convert to numpy array
ratings_train = torch.cat(ratings_train).numpy()  # concatenate batches and convert to numpy array
ratings_test = torch.cat(ratings_test).numpy()  # concatenate batches and convert to numpy array

plot_and_r2(preds_train, preds_test, ratings_train, ratings_test, advisor)




