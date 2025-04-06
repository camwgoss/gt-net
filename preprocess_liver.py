import os
from PIL import Image
import numpy as np

import utils.image_processing as image_processing
import utils.split_data as split_data


def download_and_preprocess_data(output_size: int = 256, output_type='crop'):
    '''
    Download and preprocess the brain tumor segmentation dataset from
    https://www.kaggle.com/datasets/atikaakter11/brain-tumor-segmentation-dataset
    Images will be converted to grayscale

    Parameters
        output_size: These images are large and can take a long time to train 
        on. Select output size in pixels for the square output.
        output_type:
            'crop': Randomly sample a square section of each image/mask.
            Must be smaller than the smallest dimension of the raw images.
            'resize': Resize the original image. This will apply a uniform
            scaling and crop as necessary to create a square output.
    '''
    
    repo_dir = os.path.dirname(__file__)
    raw_path = os.path.join(repo_dir, 'data','liver_tumor_segmentation','raw')
    print(raw_path)

    images, masks = _get_images_masks(raw_path)
    fig = image_processing.plot_images_masks(images, masks)
    save_path = os.path.join(
        repo_dir, 'data', 'liver_tumor_segmentation', 'raw')
    fig.savefig(save_path, bbox_inches='tight')

    if output_type == 'crop':
        images, masks = image_processing.crop_images(
            images, masks, output_size)
    elif output_type == 'resize':
        images, masks = image_processing.resize_images(
            images, masks, output_size)

    fig = image_processing.plot_images_masks(images, masks)
    repo_dir = os.path.dirname(__file__)
    save_path = os.path.join(
        repo_dir, 'data', 'liver_tumor_segmentation', 'processed')
    fig.savefig(save_path, bbox_inches='tight')

    images_train, images_eval, images_test = split_data.split_data(images)
    masks_train, masks_eval, masks_test = split_data.split_data(masks)

    _save_processed_data(images_train, masks_train,
                         images_eval, masks_eval,
                         images_test, masks_test)


def load_processed_data():
    '''
    Returns:
        images_train, masks_train, images_eval, masks_eval, images_test, masks_test:
        List of Numpy arrays containing image and mask data, (row, column, [channel]).
    '''

    repo_dir = os.path.dirname(__file__)
    data_path = os.path.join(
        repo_dir, 'data', 'liver_tumor_segmentation', 'processed_data.npz')
    processed_data = np.load(data_path)
    return processed_data

def _save_processed_data(images_train, masks_train,
                         images_eval, masks_eval,
                         images_test, masks_test):

    processed_data = [images_train, masks_train,
                      images_eval, masks_eval,
                      images_test, masks_test]

    repo_dir = os.path.dirname(__file__)
    save_path = os.path.join(
        repo_dir, 'data', 'liver_tumor_segmentation', 'processed_data.npz')

    np.savez(save_path, *processed_data)

def _get_images_masks(path: str):
    '''
    Get image and mask data for all tumor types. This processing step tosses
    information about tumor type, so the mask only contains labels for no tumor
    (0) and tumor (255).
    Arguments:
        path: Path to raw downloaded data.
    Returns:
        images, masks: List of Numpy arrays, (row, column, [channel])
    '''

    # used to store data for all tumor types
    images = []
    image_files = []
    masks = []
    mask_files = []

    image_dir = os.path.join(path,'image')
    images_subset, image_files_subset = image_processing.get_images(
        image_dir)
    images += images_subset
    image_files += image_files_subset

    mask_dir = os.path.join(path,'mask')
    masks_subset, mask_files_subset = image_processing.get_images(mask_dir)
    masks += masks_subset
    mask_files += mask_files_subset

    return images, masks


if __name__ == '__main__':
    download_and_preprocess_data(output_size=256, output_type='resize')

