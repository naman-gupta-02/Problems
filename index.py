import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import interp1d

# Step 1: Regularize Shapes
def regularize_shapes(path_data):
    n_points = len(path_data)
    
    # Calculate pairwise distances between points
    dists = pdist(path_data)
    dist_matrix = squareform(dists)
    
    # Identify circles based on distance from centroid
    center = np.mean(path_data, axis=0)
    distances_to_center = np.linalg.norm(path_data - center, axis=1)
    is_circle = np.allclose(distances_to_center, distances_to_center.mean(), rtol=0.1)
    
    if is_circle:
        return "Circle", center, distances_to_center.mean()
    
    # For now, returning "Unknown" for other shapes
    return "Unknown", None, None

# Step 2: Symmetry Exploration
def check_reflective_symmetry(path_data):
    n_points = len(path_data)
    
    # Check vertical symmetry (x-axis)
    mirrored_x = np.copy(path_data)
    mirrored_x[:, 0] = -mirrored_x[:, 0]
    is_symmetric_x = np.allclose(mirrored_x, path_data[::-1], atol=1e-2)
    
    # Check horizontal symmetry (y-axis)
    mirrored_y = np.copy(path_data)
    mirrored_y[:, 1] = -mirrored_y[:, 1]
    is_symmetric_y = np.allclose(mirrored_y, path_data[::-1], atol=1e-2)
    
    return is_symmetric_x, is_symmetric_y

# Step 3: Curve Completion
def complete_curve(path_data):
    path_data = path_data[np.argsort(path_data[:, 0])]
    x = path_data[:, 0]
    y = path_data[:, 1]
    
    interp_func = interp1d(x, y, kind='linear', fill_value="extrapolate")
    x_new = np.linspace(x.min(), x.max(), num=500)
    y_new = interp_func(x_new)
    
    return x_new, y_new

# Step 4: Visualizing and Comparing Results
def visualize_comparison(csv_path, svg_path=None):
    data = np.genfromtxt(csv_path, delimiter=',')
    unique_paths = np.unique(data[:, 0])
    
    for path_id in unique_paths:
        path_data = data[data[:, 0] == path_id][:, 1:]
        
        shape, center, radius = regularize_shapes(path_data)
        sym_x, sym_y = check_reflective_symmetry(path_data)
        x_new, y_new = complete_curve(path_data)
        
        # Visualization
        plt.figure(figsize=(8, 6))
        plt.plot(path_data[:, 0], path_data[:, 1], label='Original Curve', linewidth=2)
        plt.plot(x_new, y_new, 'r--', label='Completed Curve', linewidth=2)
        
        if shape == "Circle":
            circle = plt.Circle(center, radius, color='g', fill=False, linewidth=2, label='Regularized Circle')
            plt.gca().add_patch(circle)
        
        plt.title(f"Path ID: {path_id} | Shape: {shape} | Symmetry X: {sym_x}, Y: {sym_y}")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.show()
        
        if svg_path:
            print(f"Compare with expected SVG: {svg_path}")

# Example usage:
csv_file_path = 'occlusion2.csv'  # Update with actual CSV file path
svg_file_path = 'occlusion2.svg'  # Update with actual SVG file path if available
visualize_comparison(csv_file_path, svg_file_path)