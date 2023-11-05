import numpy as np

## Import your (n x 7 x 7) tensor here.
num_sample = 100
grids_stack = np.random.choice(np.arange(5), size = (num_sample, 7, 7)) #Randomly Sample Grids

num_sample = grids_stack.shape[0]
## Get masking of samples. 

mask = np.random.choice(a = np.arange(start = 0, stop = num_sample, step = 1), size = num_sample, replace = True)

selected_grids = grids_stack[mask]

np.save('selected_grids.npy', selected_grids)
