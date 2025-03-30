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


def threshold_masks(masks: list):
    '''
    When masks are stored using lossy compression (e.g., JPEG), they take on
    values other than 0 and 255. Use thresholding to convert to binary masks.
    This operation is performed in place.
    Arguments:
        masks: List of Numpy arrays, (image, row, column, [channel]).
    Returns:
        masks: Thresholded masks, (image, row, column, [channel]).
    '''

    for mask in masks:
        mask[mask < 128] = 0
        mask[mask >= 128] = 255
    return masks


def crop_images(images: list, masks: list = None, crop_size: int = 256):
    '''
    Randomly extract a square section from each image. If the image has a 
    dimension smaller than section_size, that image will be removed.
    Arguments:
        images: List of Numpy arrays, (image, row, column, [channel]).
        masks: Must have identical dimensions to images.
        crop_size: Dimension in pixels of square section to extract.
    Returns:
        image_sections: List of Numpy arrays containing image sections.
        mask_sections: This will only be returned if masks were provided.
    '''

    image_sections = []
    mask_sections = []

    for ii in range(len(images)):

        image = images[ii]
        rows = image.shape[0]
        cols = image.shape[1]

        if crop_size > rows or crop_size > cols:
            pass  # the section is larger than the image, so toss
        else:
            row_start = np.random.randint(rows - crop_size + 1)
            row_end = row_start + crop_size

            col_start = np.random.randint(cols - crop_size + 1)
            col_end = col_start + crop_size

            image_section = image[row_start:row_end, col_start:col_end]
            image_sections.append(image_section)

            if masks is not None:
                mask = masks[ii]
                mask_section = mask[row_start:row_end, col_start:col_end]
                mask_sections.append(mask_section)

    if masks is None:
        return image_sections
    else:
        return image_sections, mask_sections
