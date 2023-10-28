import numpy as np
from skimage import measure

def get_cluster_boundary(grid, target_value):
    # Convert the grid to a NumPy array
    grid_array = np.array(grid)

    # Create a binary mask for the target cluster
    cluster_mask = (grid_array == target_value).astype(int)

    # Find contours using the marching squares algorithm
    contours = measure.find_contours(cluster_mask, 0.5)

    # Select the longest contour as the boundary
    boundary = max(contours, key=len)

    # Convert boundary coordinates to (x, y) format
    boundary_points = [(int(point[1]), int(point[0])) for point in boundary]

    return boundary_points

grid = [[1, 2, 2, 2, 1, 4, 2],
        [1, 0, 1, 1, 2, 4, 2],
        [2, 3, 4, 2, 0, 1, 2],
        [1, 2, 2, 2, 2, 2, 3],
        [1, 1, 1, 1, 1, 2, 2],
        [1, 1, 1, 1, 4, 4, 4],
        [1, 2, 2, 3, 4, 3, 3]]

target_value = 2  # The cluster you want to find the boundary for

boundary_points = get_cluster_boundary(grid, target_value)

print("Boundary Points for Cluster", target_value, ":", boundary_points)