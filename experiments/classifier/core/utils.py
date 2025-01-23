import json
import os
from PIL import Image
from multiprocessing import Pool, cpu_count



def create_task_dict(dataset_path):
    """
    Creates a dictionary of tasks from the given dataset path.
    Args:
        dataset_path (str): The path to the dataset directory.
    Returns:
        dict: A dictionary where the keys are task names and the values are lists of task numbers.
    """
    
    task_dict = {}
    folders = os.listdir(dataset_path)
    folders.sort()
    tasks = [(folder.split('_')[1], folder.split('_')[2]) for folder in folders if len(folder.split('_')) == 3]
    for task_number, task_name in tasks:
        if task_name not in task_dict:
            task_dict[task_name] = []
        task_dict[task_name].append(task_number)
    return task_dict


def get_data_path(dataset_path, task_dict, task_names, init_term):
    """
    Retrieves the paths and labels for positive and negative data samples from a dataset.
    Args:
        dataset_path (str): The base path to the dataset.
        task_dict (dict): A dictionary where keys are task names and values are lists of task numbers.
        task_names (str or list): A single task name or a list of task names to search for.
        init_term (str): The initial term to append to the folder path. Typically "init" or "term".
    Returns:
        tuple: A tuple containing four lists:
            - positive_data_paths (list): List of file paths to positive data images.
            - negative_data_paths (list): List of file paths to negative data images.
            - positive_labels (list): List of labels corresponding to positive data images. This mean the labeled points, etc.
            - negative_labels (list): List of labels corresponding to negative data images.
    """
    
    if not isinstance(task_names, list):
        task_names = [task_names]

    matching_folders = []
    for task_name in task_names:
        for task_number in task_dict.get(task_name, []):
            folder_name = f"run_{task_number}_{task_name}"
            folder_path = os.path.join(dataset_path, folder_name, init_term)
            if os.path.exists(folder_path):
                matching_folders.append(folder_path)
    
    positive_data_paths = []
    positive_labels = []
    negative_data_paths = []
    negative_labels = []

    for folder in matching_folders:
        pos_folder = os.path.join(folder, 'positive')
        neg_folder = os.path.join(folder, 'negative')

        # Load positive images and labels
        pos_images = sorted(os.listdir(pos_folder))
        with open(os.path.join(pos_folder, 'labels.json'), 'r') as f:
            pos_labels = json.load(f)
        for img in pos_images:
            if img in pos_labels:
                positive_data_paths.append(os.path.join(pos_folder, img))
                positive_labels.append(pos_labels[img])

        # Load negative images and labels
        neg_images = sorted(os.listdir(neg_folder))
        with open(os.path.join(neg_folder, 'labels.json'), 'r') as f:
            neg_labels = json.load(f)
        for img in neg_images:
            if img in neg_labels:
                negative_data_paths.append(os.path.join(neg_folder, img))
                negative_labels.append(neg_labels[img])

    return positive_data_paths, negative_data_paths, positive_labels, negative_labels


def verify_image(file_path):
    """
    Verifies if the given file path points to a valid image file.

    This function attempts to open the image file and verify its integrity.
    If the file is a valid image, the file path is returned. If the file is
    not a valid image or an error occurs during the verification process,
    None is returned.

    Args:
        file_path (str): The path to the image file to be verified.

    Returns:
        str or None: The file path if the image is valid, otherwise None.

    Raises:
        OSError: If an OS-related error occurs.
        IOError: If an input/output error occurs.
        SyntaxError: If the file is not a valid image.
    """
    try:
        with Image.open(file_path) as img:
            img.verify()  
        return file_path 
    except (OSError,IOError,SyntaxError):
        return None


def filter_valid_images(file_paths):
    """
    Filters a list of image file paths, returning only the valid ones.

    This function uses multiprocessing to verify the validity of each image file path
    in the provided list. It returns a list of valid image file paths.

    Args:
        file_paths (list): A list of file paths to be verified.

    Returns:
        list: A list of valid image file paths.
    """
    with Pool(cpu_count()) as pool:
        valid_files = pool.map(verify_image, file_paths)
    return [file for file in valid_files if file is not None]