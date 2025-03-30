import os
from PIL import Image
import numpy as np


def get_images(image_dir: str, grayscale=True):
    '''
    Get all the images in a directory. Images must have the same dimensions.
    Arguments:
        image_dir: Directory containing images.
        grayscale: whether to return images in grayscale.
    Returns:
        images: List of Numpy arrays, (image, row, column, [channel]).
        image_files: List of strings containing file names.
    '''

    image_files = os.listdir(image_dir)
    image_files.sort()

    images = []
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path)

        if grayscale == True:
            image = image.convert('L')  # 'L' converts to grayscale

        image = np.array(image)
        images.append(image)
        
    return images, image_files


def threshold_masks(masks: np.array):
    '''
    When masks are stored using lossy compression (e.g., JPEG), they take on
    values other than 0 and 255. Use thresholding to convert to binary masks.
    This operation is performed in place.
    Arguments:
        masks: Image data, (image, row, column, [channel]).
    Returns:
        masks: Thresholded masks, (image, row, column, [channel]).
    '''
    
    for mask in masks:
        mask[mask < 128] = 0
        mask[mask >= 128] = 255
    return masks
