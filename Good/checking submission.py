from utils_public import *
import numpy as np
import pdb

original_grid = np.load('000995178.npy')
bad_grid = np.load('grid85.npy').astype(int)
print(bad_grid)
for i in range(100):
    if np.all(original_grid[i] == bad_grid):
        print(i)
        print(original_grid[i])
    
