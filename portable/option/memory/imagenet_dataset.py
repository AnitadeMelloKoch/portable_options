import os
import numpy as np
from PIL import Image
from torchvision import transforms
import xml.etree.ElementTree as ET  # XML parser

# Define paths for images and annotations
def pad_func(x):
    return x

class ImageNetDataset:
    def __init__(
        self,
        batchsize=16,
        unlabelled_batchsize=None,
        max_size=100000,
        pad_func=pad_func,
        create_validation_set=False,
        store_int=True,
    ):
        self.batchsize = batchsize
        self.data_batchsize = batchsize // 2
        self.pad = pad_func
        self.counter = 0
        self.num_batches = 0
        self.list_max_size = max_size // 2
        self.store_int = store_int


# Define folders for Chihuahua and Spaniel images
chihuahua_image_dir = 'Images/n02085620-Chihuahua'  # Chihuahua images folder
spaniel_image_dir = 'Images/n02085782-Japanese_spaniel'  # Spaniel images folder

# Function to load images from a given folder
def load_images_from_folder(folder, transform=None):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            try:
                image = Image.open(img_path).convert('RGB')  # Convert image to RGB
                if transform:
                    image = transform(image)
                images.append(np.array(image))  # Convert image to numpy array
                filenames.append(filename)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return images, filenames

# Function to load XML annotations
def load_annotations_from_folder(folder):
    annotations = []
    filenames = []
    for root, _, files in os.walk(folder):  # Traverse folder and subfolders
        for filename in files:
            file_path = os.path.join(root, filename)
            # Assuming annotation files are in XML format
            if filename.lower().endswith('.xml'):
                try:
                    tree = ET.parse(file_path)  # Parse the XML file
                    root_element = tree.getroot()
                    # You can extract specific data here based on the XML structure
                    annotations.append(ET.tostring(root_element, encoding='unicode'))  # Store XML content as string
                    filenames.append(filename)
                except Exception as e:
                    print(f"Error loading annotation {file_path}: {e}")
            else:
                print(f"Skipped non-XML file: {filename}")
    if not annotations:
        print(f"No annotations found in folder: {folder}")
    return annotations, filenames

# Define any image transformations you want (e.g., resizing)
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 256x256
    transforms.ToTensor(),  # Convert to tensor
])

# Load the images for Chihuahuas and Spaniels
chihuahua_images, chihuahua_filenames = load_images_from_folder(chihuahua_image_dir, transform=image_transform)
spaniel_images, spaniel_filenames = load_images_from_folder(spaniel_image_dir, transform=image_transform)

# Print shape of each Chihuahua image after preprocessing
print("Chihuahua images:")
for img, filename in zip(chihuahua_images, chihuahua_filenames):
    print(f"Image {filename} shape after preprocessing: {img.shape}")

# Print shape of each Spaniel image after preprocessing
print("Spaniel images:")
for img, filename in zip(spaniel_images, spaniel_filenames):
    print(f"Image {filename} shape after preprocessing: {img.shape}")

# Specify where to save the images and annotations
save_dir = '/home/yyang239/divdis/portable_options/resources/dog_images'
os.makedirs(save_dir, exist_ok=True)

# Save each Chihuahua image as an individual .npy file
for img, filename in zip(chihuahua_images, chihuahua_filenames):
    img_save_path = os.path.join(save_dir, f'chihuahua_{os.path.splitext(filename)[0]}.npy')
    np.save(img_save_path, img)
    print(f"Saved Chihuahua image: {img_save_path}")

# Save each Spaniel image as an individual .npy file
for img, filename in zip(spaniel_images, spaniel_filenames):
    img_save_path = os.path.join(save_dir, f'spaniel_{os.path.splitext(filename)[0]}.npy')
    np.save(img_save_path, img)
    print(f"Saved Spaniel image: {img_save_path}")

print("All images saved individually.")

