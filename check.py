import numpy as np

def print_npy_shape(file_path):
    try:
        image_array = np.load(file_path)
        print(f"{file_path}: {image_array.shape}")  # Shape of the NumPy array
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

# Example usage
npy_file = "/home/yyang239/portable_options/resources/dog_images/spaniel_n02085782_4798.npy"
print_npy_shape(npy_file)
