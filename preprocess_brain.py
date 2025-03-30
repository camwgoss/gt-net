import kagglehub
import os
from PIL import Image
import numpy as np

import utils.image_processing as image_processing


def download_and_preprocess_data(crop_size: int = 256):
    '''
    Download and preprocess the brain tumor segmentation dataset from 
    https://www.kaggle.com/datasets/atikaakter11/brain-tumor-segmentation-dataset
    Images will be converted to grayscale.

    Parameters
        crop_size: These images are large and can take a long time to
        train on. Provide a section size in pixels to randomly sample a
        square section of each image/mask to speed up training. Must be smaller
        than the smallest dimension of the raw images.
    Returns
        train_path, eval_path, test_path: Processed data paths.
    '''

    raw_path = _download_data()

    images, masks = _get_images_masks(raw_path)

    images, masks = image_processing.crop_images(
        images, masks, crop_size)

    # TODO divide images/masks into training, evaluation, and test sets


def _download_data():
    '''
    Download brain tumor segmetnation dataset.
    Returns
        path: Path to raw downloaded data
    '''

    repo_dir = os.path.dirname(__file__)
    data_dir = os.path.join(
        repo_dir, 'data', 'brain_tumor_segmentation', 'raw')
    os.makedirs(data_dir, exist_ok=True)

    # data downloads to KAGGLEHUB_CACHE by default; change to ./data
    os.environ['KAGGLEHUB_CACHE'] = data_dir

    # brain tumor segmentation dataset from
    # https://www.kaggle.com/datasets/atikaakter11/brain-tumor-segmentation-dataset
    path = kagglehub.dataset_download(
        "atikaakter11/brain-tumor-segmentation-dataset")
    return path


def _get_images_masks(path: str):
    '''
    Get image and mask data for all tumor types. This processing step tosses
    information about tumor type, so the mask only contains labels for no tumor
    (0) and tumor (255).
    Arguments:
        path: Path to raw downloaded data.
    Returns:
        images, masks: Numpy array of image data, (image, row, column, [channel])
    '''

    # used to store data for all tumor types
    images = []
    image_files = []
    masks = []
    mask_files = []

    datasets = [0, 1, 2, 3]  # 0: none, 1: glioma, 2: meningioma, 3: pituitary
    for dataset in datasets:
        print('dataset', dataset)
        image_dir = os.path.join(
            path, 'Brain Tumor Segmentation Dataset', 'image', str(dataset))
        images_subset, image_files_subset = image_processing.get_images(
            image_dir)
        images += images_subset
        image_files += image_files_subset

        mask_dir = os.path.join(
            path, 'Brain Tumor Segmentation Dataset', 'mask', str(dataset))
        masks_subset, mask_files_subset = image_processing.get_images(mask_dir)
        masks += masks_subset
        mask_files += mask_files_subset

    # This is not a perfect dataset, so there are a couple images without
    # corresponding masks or vice versa. Crawl through masks and images and
    # toss un-paired entries.

    # crawl images in reverse order to make item removal easier
    for ii in np.flip(np.arange(len(image_files))):
        image_file = image_files[ii]
        root, ext = os.path.splitext(image_file)
        mask_file = root + '_m' + ext  # mask files add '_m' to file name
        if mask_file not in mask_files:  # no corresponding mask
            print(mask_file)
            images.pop(ii)
            image_files.pop(ii)

    # crawl masks in reverse order to make item removal easier
    for mm in np.flip(np.arange(len(mask_files))):
        mask_file = mask_files[mm]
        root, ext = os.path.splitext(mask_file)
        image_file = root[:-2] + ext  # remove '_m' from mask file name
        if image_file not in image_files:  # no corresponding image, so toss
            print(image_file)
            masks.pop(mm)
            mask_files.pop(mm)

    # resize masks to match image dimensions
    for ii in range(len(images)):
        image = images[ii]
        image = Image.fromarray(image)

        mask = masks[ii]
        mask = Image.fromarray(mask)  # np.array -> Image
        mask = mask.resize(image.size)  # resize mask to image dimensions
        mask = np.array(mask)  # Image -> np.array
        masks[ii] = mask

    masks = image_processing.threshold_masks(masks)

    return images, masks


if __name__ == '__main__':
    download_and_preprocess_data(crop_size=256)
