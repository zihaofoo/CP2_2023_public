from utils_public import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import pdb 
from torch.utils.data import DataLoader
# import torch.nn.functional as F
import random

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




def fit_plot_predict_zyzh(grids, ratings, i):
    """ 
    Function to implement autoML to correlate test data and labels, for a given advisor i
    Returns the Prediction (model y-values) and the Predictor (Trained Model)
    """
    random.seed(42)
    class CNNModel(nn.Module):
        def __init__(self, num_classes):
            super(CNNModel, self).__init__()
            self.conv1 = nn.Conv2d(5, 50, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)  # MaxPooling to reduce spatial dimensions
            self.conv2 = nn.Conv2d(50, 15, kernel_size=3, padding=1)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)  # MaxPooling to reduce spatial dimensions
            self.fc1 = nn.Linear(735, 128)  # Adjust input size based on output size of previous layer
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.relu(x)
            x = x.view(x.size(0), -1)  # Flatten the output
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    grids_subset, ratings_subset = select_rated_subset(grids, ratings[:,i])
    grids_subset = grids_subset.astype(int)
    one_hot_encoded_grids = np.eye(5)[grids_subset]
    # Convert one_hot_encoded_grids to PyTorch tensor
    one_hot_encoded_grids = torch.Tensor(one_hot_encoded_grids)
    
    preds_train_df = pd.DataFrame(ratings_subset, columns=["ratings"])

    # Create an instance of the model
    model = CNNModel(num_classes=1)

    # Define a loss function and an optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 300
    batch_size = 128


    one_hot_encoded_grids=one_hot_encoded_grids.permute(0, 3, 1, 2)
    feats_train, feats_test, ratings_train, ratings_test = train_test_split(one_hot_encoded_grids, ratings_subset)

    # Create data loaders for training and testing data
    train_loader = DataLoader(dataset=list(zip(feats_train, ratings_train)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=list(zip(feats_test, ratings_test)), batch_size=batch_size)    
    # preds_train = pd.DataFrame(ratings_train, columns = ["ratings"])

    ## all_train = pd.concat([feats_train, ratings_train], axis=1)
    
    for epoch in range(num_epochs):
        model.train()
        
        for data, labels in train_loader:
            
            optimizer.zero_grad()
            outputs = model(data)
            labels = labels.unsqueeze(1)
            loss = criterion(outputs.float(), labels.float())
            loss.backward()
            optimizer.step()

    # Make predictions using your CNN model
    model.eval()
    preds_train = []
    preds_test = []
    with torch.no_grad():
        for data, labels in train_loader:
            outputs = model(data.float())
            preds_train.extend(outputs)
        for data, labels in test_loader:
            outputs = model(data.float())
            preds_test.extend(outputs)

    r2_train = r2_score(ratings_train, preds_train)
    r2_test = r2_score(ratings_test, preds_test)
    
    print(f"Train Set R2 score: {r2_train}")
    print(f"Test Set R2 score: {r2_test}")

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
