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
import random

## Setting random seeds
random_seed = 42
random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)

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
grids_subset_tensor = torch.tensor(grids_subset, dtype = torch.float32)
ratings_subset_tensor = torch.tensor(ratings_subset, dtype = torch.float32)

# Splitting data into train and test
test_train_split = 0.8
dataset = TensorDataset(grids_subset_tensor, ratings_subset_tensor)
train_size = int(test_train_split * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size = 32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN Model Definition
class CNNModel(nn.Module):
    def __init__(self, filter_size:int, kernel_size:int, dense_units:int, dropout:float, conv_number:int, fc_number:int):
        super(CNNModel, self).__init__()

        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.dense_units = dense_units
        self.dropout = dropout
        self.conv_number = conv_number
        self.fc_number = fc_number
        self.conv1 = nn.Conv2d(in_channels = 5, out_channels = filter_size, kernel_size = kernel_size, padding = kernel_size//2)
        self.conv2 = nn.Conv2d(in_channels = filter_size, out_channels = filter_size * 2, kernel_size = kernel_size, padding = kernel_size//2)
        
        # self.dropout = nn.Dropout(p=dropout)
    
        # Use a dummy input to calculate the correct size after conv layers
        dummy_input = torch.zeros(1, 5, 7, 7)  # Assuming the input size is (batch, channels, height, width)
        with torch.no_grad():
            dummy_output = self.conv2(self.conv1(dummy_input))
            
        num_flatten_features = dummy_output.numel()
        self.fc1 = nn.Linear(in_features = num_flatten_features, out_features = dense_units)
        self.fc2 = nn.Linear(in_features = dense_units, out_features = 1)  # Change 10 to the number of classes you have

    # Dynamically create different convolution layers
    def create_conv_layers(self, conv_number):
        layers = []
        in_channels = 5
        
        for _ in range(conv_number):
            conv_layer = nn.Conv2d(in_channels, self.filter_size, self.kernel_size, padding=self.kernel_size // 2)
            layers.append(conv_layer)
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.ReLU())
            in_channels = self.filter_size
        return nn.Sequential(*layers)

    def calculate_output_size(self, input_size = (7,7)):
        """
        Calculate the output size of a feature map after a series of convolutional layers.
        Args:
        - input_size: Tuple representing the input size (height, width).
        Returns:
        - size of output
        """
        output_size = input_size  # Start with the input size

        for layer in range(self.conv_number):
            kernel_size = self.kernel_size
            padding = self.kernel_size // 2
            stride = 1

            # Calculate output size for the current layer
            output_size = (
                ((output_size[0] - kernel_size + 2 * padding) // stride) + 1,
                ((output_size[1] - kernel_size + 2 * padding) // stride) + 1
            )
        
        
        size = output_size[0] * output_size[1]

        return size

    # Dynamically create different fc layers
    def create_fc_layers(self, fc_number):
        layers = []
        in_features = self.filter_size * self.calculate_output_size()
        for _ in range(fc_number):
            fc_layer = nn.Linear(in_features, self.dense_units)
            layers.append(fc_layer)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            in_features = self.dense_units
        # Remove the last dropout layer
        layers.pop()
        return nn.Sequential(*layers)
    


    def forward(self, x, conv_number, fc_number):

        conv_layers = self.create_conv_layers(conv_number)
        x = conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        fc_layers = self.create_fc_layers(fc_number)        
        x = fc_layers(x)
        x = torch.sigmoid(x)  # Sigmoid activation for the output layer
        return x

# Define the objective function for Bayesian Optimization
def objective_function(params):
    
    filter_size = int(params[0, 0])
    kernel_size = int(params[0, 1])
    dense_units = int(params[0, 2])
    learning_rate = params[0, 3]
    weight_decay = params[0, 4]
    epochs = int(params[0, 5])
    dropout = params[0, 6]
    conv_number = int(params[0, 7])
    fc_number = int(params[0, 8])
    

    model = CNNModel(filter_size, kernel_size, dense_units,dropout,conv_number,fc_number).to(device)
    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data,conv_number,fc_number)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
    
    # Validation loop (assuming you have a validation set)
    model.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data,conv_number,fc_number)
            loss = criterion(outputs, target)
            validation_loss += loss.item() * data.size(0)
    validation_loss /= len(test_loader.dataset)
    
    # Return the validation loss as the objective to minimize
    return validation_loss

# Bayesian optimization bounds
bounds = [
    {'name': 'filter_size', 'type': 'discrete', 'domain': (32, 128)},
    {'name': 'kernel_size', 'type': 'discrete', 'domain': (2, 5)},
    {'name': 'dense_units', 'type': 'discrete', 'domain': (32, 256)},
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-4, 1e-3)},
    {'name': 'weight_decay', 'type': 'continuous', 'domain': (1e-4, 1e-2)},
    {'name': 'epochs', 'type': 'discrete', 'domain': (10, 150)},
    {'name': 'dropout', 'type': 'continuous', 'domain': (0.5, 0.8)},
    {'name': 'conv_number', 'type': 'discrete', 'domain': (2, 5)},
    {'name': 'fc_number', 'type': 'discrete', 'domain': (2, 5)},
    
    
]

# Initialize Bayesian Optimization
optimizer = GPyOpt.methods.BayesianOptimization(f = objective_function, domain = bounds, verbosity = True)
# 
# # Start the optimization process
optimizer.run_optimization(max_iter = 5)
# 
# # Print the best hyperparameters
print("Best hyperparameters:")
print(f"Filter size: {int(optimizer.x_opt[0])}")
print(f"Kernel size: {int(optimizer.x_opt[1])}")
print(f"Dense units: {int(optimizer.x_opt[2])}")
print(f"Learning rate: {optimizer.x_opt[3]}")
print(f"Weight Decay: {optimizer.x_opt[4]}")
print(f"Epoch: {optimizer.x_opt[5]}")
print(f"Dropout: {optimizer.x_opt[6]}")
print(f"Number of Convolution: {optimizer.x_opt[7]}")
print(f"Number of fully connected: {optimizer.x_opt[8]}")
print(f"Best validation loss: {optimizer.fx_opt}")


# Extract the best hyperparameters
best_filter_size = int(optimizer.x_opt[0])
best_kernel_size = int(optimizer.x_opt[1])
best_dense_units = int(optimizer.x_opt[2])
best_learning_rate = optimizer.x_opt[3]
best_weight_decay = optimizer.x_opt[4]
best_epochs = int(optimizer.x_opt[5])
best_dropout = (optimizer.x_opt[6])
best_conv_number = int(optimizer.x_opt[7])
best_fc_number = int(optimizer.x_opt[8])


# Optimized hyperparameters for CNN
##best_filter_size = 64
##best_kernel_size = 3
##best_dense_units = 128
##best_learning_rate = 1E-3
##best_weight_decay = 1E-3
##best_epochs = 10

# Rebuild the model with the best hyperparameters
final_model = CNNModel(best_filter_size, best_kernel_size, best_dense_units, best_dropout, best_conv_number, best_fc_number).to(device)

# Loss and Optimizer for the final model
criterion = nn.MSELoss()
optimizer = optim.Adam(final_model.parameters(), lr = best_learning_rate, weight_decay = best_weight_decay)

# Training the final model
epochs = best_epochs  # Or however many epochs you deem necessary
for epoch in range(epochs):
    final_model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = final_model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)

    train_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")

# Evaluate the final model on test data
final_model.eval()
test_loss = 0.0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = final_model(data)
        loss = criterion(outputs, target)
        test_loss += loss.item() * data.size(0)
test_loss /= len(test_loader.dataset)
print(f"Test Loss: {test_loss:.4f}")

# Predictions for plotting and R^2 calculation
preds_train = []
preds_test = []
ratings_train = []
ratings_test = []

with torch.no_grad():
    for data, target in DataLoader(train_dataset, batch_size=32):
        data = data.to(device)
        outputs = final_model(data)
        preds_train.append(outputs.cpu())
        ratings_train.append(target)

    for data, target in DataLoader(test_dataset, batch_size=32):
        data = data.to(device)
        outputs = final_model(data)
        preds_test.append(outputs.cpu())
        ratings_test.append(target)

# Concatenate all the batch results
preds_train = torch.cat(preds_train).numpy()
preds_test = torch.cat(preds_test).numpy()
ratings_train = torch.cat(ratings_train).numpy()
ratings_test = torch.cat(ratings_test).numpy()

# Save the entire model
torch.save(final_model, 'final_model' + np.str_(advisor) + '.pth')

# Later on, to load the entire model
loaded_model = torch.load('final_model' + np.str_(advisor) + '.pth')
loaded_model.eval()  # Don't forget to call eval() for inference

# Plot the results and calculate R^2 scores
plot_and_r2(preds_train, preds_test, ratings_train, ratings_test, advisor)






