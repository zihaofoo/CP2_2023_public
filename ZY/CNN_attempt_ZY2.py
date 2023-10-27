from utils_public import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from autogluon.tabular import TabularPredictor
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import pdb 
from torch.utils.data import DataLoader


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



# Define your CNN model for regression
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(5, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten the output for fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load your data (grids and ratings)
grids = load_grids()
ratings = np.load("datasets/scores.npy")

# Specify the number of classes (regression target)
num_classes = 1

# Split the data into training and testing sets
feats_train, feats_test, ratings_train, ratings_test = train_test_split(grids, ratings[:, 2], test_size=0.2)

# Convert data to PyTorch tensors
feats_train = torch.Tensor(feats_train)
feats_test = torch.Tensor(feats_test)
ratings_train = torch.Tensor(ratings_train)
ratings_test = torch.Tensor(ratings_test)

# Create a DataLoader for training and testing data
batch_size = 32
train_loader = DataLoader(dataset=list(zip(feats_train, ratings_train)), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=list(zip(feats_test, ratings_test)), batch_size=batch_size)

# Create an instance of the model
model = CNNModel(num_classes)

# Define a loss function and an optimizer for regression
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    for data, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
preds_train = []
preds_test = []

with torch.no_grad():
    for data, labels in train_loader:
        outputs = model(data)
        preds_train.extend(outputs)

    for data, labels in test_loader:
        outputs = model(data)
        preds_test.extend(outputs)

# Calculate R-squared for training and testing
r2_train = r2_score(ratings_train, preds_train)
r2_test = r2_score(ratings_test, preds_test)

print(f"Train Set R2 score: {r2_train}")
print(f"Test Set R2 score: {r2_test}")