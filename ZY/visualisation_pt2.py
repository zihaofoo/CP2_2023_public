
import numpy as np
import pandas as pd
from utils_public import *

grids_subset = np.zeros((1250, 7, 7))
# Load grids from 'grids.npz' using a context manager
with np.load('grids.npz') as data:
    grids_subset = data['arr_0']
ground_truth_df = pd.read_csv('ground_truth_nic.csv',header=None)
ground_truth = np.array(ground_truth_df)

prediction_df = pd.read_csv('prediction_nic.csv',header=None)
prediction = np.array(prediction_df)
difference = prediction - ground_truth

# What I want to see 
# Mask the differences where ground truth > 0.85
mask = ground_truth > 0.85
filtered_differences = difference[mask]
filtered_ground_truth = ground_truth[mask]
filtered_prediction = prediction[mask]
# Create an array of indices where the mask is True
filtered_indices = np.where(mask)

# Use advanced indexing to filter grids_subset
filtered_grids_subset = grids_subset[filtered_indices[0]]

# Sort the filtered differences
sorted_indices = np.argsort(filtered_differences)
print(sorted_indices)

# Get the top 20 and bottom 20 indices
top_20_indices = sorted_indices[-20:]
bottom_20_indices = sorted_indices[:20]
top_20_grids = filtered_grids_subset[top_20_indices]
bottom_20_grids = filtered_grids_subset[bottom_20_indices]

# Create DataFrames for the top 20 and bottom 20 ground truth and prediction
top_20_ground_truth_df = pd.DataFrame(filtered_ground_truth[top_20_indices])
top_20_prediction_df = pd.DataFrame(filtered_prediction[top_20_indices])

bottom_20_ground_truth_df = pd.DataFrame(filtered_ground_truth[bottom_20_indices])
bottom_20_prediction_df = pd.DataFrame(filtered_prediction[bottom_20_indices])

# Concatenate DataFrames for top and bottom 20
combined_df = pd.concat([pd.concat([top_20_ground_truth_df, bottom_20_ground_truth_df], axis=0), pd.concat([top_20_prediction_df, bottom_20_prediction_df], axis=0)], axis=1)

# Save the combined DataFrame to a CSV file
combined_df.to_csv('combined_data.csv', index=False, header=False)

plot_n_grids(top_20_grids)

plot_n_grids(bottom_20_grids)