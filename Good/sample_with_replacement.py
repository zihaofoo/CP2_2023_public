import numpy as np
from utils_public import *

np.random.seed(42)
## Import your (n x 7 x 7) tensor here.
num_sample = 100
grids_stack = np.load('grids_best_v3.npy')
grids_size = grids_stack.shape[0]
max_score = 0
## Get masking of samples. 
for i in range(1000000):

    mask = np.random.choice(a = np.arange(start = 0, stop = grids_size, step = 1), size = num_sample, replace = True)
    final_submission = grids_stack[mask].astype(int)
    score = diversity_score(final_submission)
    if max_score < score:
        max_score = score
        print(final_submission.shape)
        print(max_score)
        assert final_submission.shape == (100, 7, 7)
        assert final_submission.dtype == int
        assert np.all(np.greater_equal(final_submission, 0) & np.less_equal(final_submission, 4))
        np.save('242424242.npy',final_submission)




