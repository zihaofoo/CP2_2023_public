import numpy as np

def max_min_distances(arr):


    mask_0 = np.argwhere(arr == 0)
    mask_1 = np.argwhere(arr == 1)
    mask_2 = np.argwhere(arr == 2)
    mask_3 = np.argwhere(arr == 3)
    mask_4 = np.argwhere(arr == 4)
    feat_out = np.zeros((1,20))
    feat_out[0, 0] = get_pairwise_dist(mask_0, mask_1)
    feat_out[0, 1] = get_pairwise_dist(mask_0, mask_2)
    feat_out[0, 2] = get_pairwise_dist(mask_0, mask_3)
    feat_out[0, 3] = get_pairwise_dist(mask_0, mask_4)
    feat_out[0, 4] = get_pairwise_dist(mask_1, mask_0)
    feat_out[0, 5] = get_pairwise_dist(mask_1, mask_2) 
    feat_out[0, 6] = get_pairwise_dist(mask_1, mask_3)
    feat_out[0, 7] = get_pairwise_dist(mask_1, mask_4)
    feat_out[0, 8] = get_pairwise_dist(mask_2, mask_0)
    feat_out[0, 9] = get_pairwise_dist(mask_2, mask_1) 
    feat_out[0, 10] = get_pairwise_dist(mask_2, mask_3)
    feat_out[0, 11] = get_pairwise_dist(mask_2, mask_4)
    feat_out[0, 12] = get_pairwise_dist(mask_3, mask_0) 
    feat_out[0, 13] = get_pairwise_dist(mask_3, mask_1)
    feat_out[0, 14] = get_pairwise_dist(mask_3, mask_2)
    feat_out[0, 15] = get_pairwise_dist(mask_3, mask_4)
    feat_out[0, 16] = get_pairwise_dist(mask_4, mask_0) 
    feat_out[0, 17] = get_pairwise_dist(mask_4, mask_1)
    feat_out[0, 18] = get_pairwise_dist(mask_4, mask_2)
    feat_out[0, 19] = get_pairwise_dist(mask_4, mask_3)
    print(feat_out)

    return 0

def get_pairwise_dist(mask_A, mask_B):

    # mask_A gives the reference points

    distance_array = np.zeros(mask_A.shape[0])
    n1 = np.shape(mask_B)[0]
    for i1 in range(np.shape(mask_A)[0]):
        dist = np.linalg.norm(mask_B - mask_A[i1] , axis = 1, ord = 2)
    #    print(dist)
        if dist.size == 0:
            return 0
        distance_array[i1] = min(dist)
    #print(distance_array)        
    if distance_array.size == 0:
        return 0

    return max(distance_array)

# Example array
arr = np.array([[4, 1, 1, 1, 2, 1, 2],
                [1, 1, 2, 1, 4, 2, 1],
                [2, 1, 1, 1, 2, 4, 3],
                [2, 1, 1, 2, 2, 1, 1],
                [1, 4, 4, 2, 1, 3, 2],
                [1, 4, 1, 3, 4, 2, 2],
                [1, 2, 1, 1, 1, 1, 1]])

nearest_distances = max_min_distances(arr)
print(nearest_distances)

