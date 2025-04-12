import kagglehub
import os
from PIL import Image
import numpy as np

import utils.image_processing as ip
import utils.split_data as split_data


def download_and_preprocess_data(output_size: int = 256, augmentation=None):
    '''
    Download and preprocess the brain tumor segmentation dataset from
    https://www.kaggle.com/datasets/atikaakter11/brain-tumor-segmentation-dataset
    Images will be converted to grayscale.

    Parameters
        output_size: These images are large and can take a long time to train 
        on. Select output size in pixels for the square output.
        augmentation: This will roughly double the dataset size by augmenting 
        it with altered images.
            'crop': Randomly sample a square section of each image/mask.
            'rotate': Randomly rotated each image/mask.
            'elastic_deformation': Random bi-cubic deformation.
    '''

    raw_path = _download_data()

    all_images_train = np.empty([0, output_size, output_size])
    all_labels_train = np.empty([0, output_size, output_size])
    all_images_val = np.empty([0, output_size, output_size])
    all_labels_val = np.empty([0, output_size, output_size])
    all_images_test = np.empty([0, output_size, output_size])
    all_labels_test = np.empty([0, output_size, output_size])

    datasets = [0, 1, 2, 3]  # 0: none, 1: glioma, 2: meningioma, 3: pituitary
    for dataset in datasets:
        images_raw, masks_raw = _get_images_masks(raw_path, dataset=dataset)

        # split immediately before any modification to avoid data contamination
        images_train_raw, images_val_raw, images_test_raw = split_data.split_data(
            images_raw)
        masks_train_raw, masks_val_raw, masks_test_raw = split_data.split_data(
            masks_raw)

        # process raw images
        images_train, masks_train = ip.resize_images(
            images_train_raw, masks_train_raw, output_size)
        images_val, masks_val = ip.resize_images(
            images_val_raw, masks_val_raw, output_size)
        images_test, masks_test = ip.resize_images(
            images_test_raw, masks_test_raw, output_size)

        # convert masks {0, 255} to labels {0, 1, 2, 3}
        labels_train = ip.masks_to_labels(
            masks_train, label=dataset)
        labels_val = ip.masks_to_labels(
            masks_val, label=dataset)
        labels_test = ip.masks_to_labels(
            masks_test, label=dataset)

        all_images_train = np.concatenate(
            [all_images_train, images_train], axis=0)
        all_labels_train = np.concatenate(
            [all_labels_train, labels_train], axis=0)
        all_images_val = np.concatenate(
            [all_images_val, images_val], axis=0)
        all_labels_val = np.concatenate(
            [all_labels_val, labels_val], axis=0)
        all_images_test = np.concatenate(
            [all_images_test, images_test], axis=0)
        all_labels_test = np.concatenate(
            [all_labels_test, labels_test], axis=0)

        if augmentation is not None:  # augment training dataset
            if augmentation == 'crop':
                processor = ip.crop_images
            elif augmentation == 'rotate':
                processor = ip.rotate_images
            elif augmentation == 'elastic_deformation':
                processor = ip.elastically_deform_images
            else:
                raise Exception('Error: Unknown augmentation provided.')

            images_train, masks_train = processor(
                images_train_raw, masks_train_raw)
            labels_train = ip.masks_to_labels(masks_train, label=dataset)

            all_images_train = np.concatenate(
                [all_images_train, images_train], axis=0)
            all_labels_train = np.concatenate(
                [all_labels_train, labels_train], axis=0)

    _save_processed_data(all_images_train, all_labels_train,
                         all_images_val, all_labels_val,
                         all_images_test, all_labels_test)

    # processed image/mask plot
    fig = ip.plot_images_labels(
        all_images_train, all_labels_train)
    repo_dir = os.path.dirname(__file__)
    save_path = os.path.join(
        repo_dir, 'data', 'brain_tumor_segmentation', 'processed')
    fig.savefig(save_path, bbox_inches='tight')


def load_processed_data():
    '''
    Returns:
        images_train, masks_train, images_val, masks_val, images_test, masks_test:
        Numpy array of image and mask data, (sample, row, column).
    '''

    repo_dir = os.path.dirname(__file__)
    data_path = os.path.join(
        repo_dir, 'data', 'brain_tumor_segmentation', 'processed_data.npz')
    processed_data = np.load(data_path)
    return processed_data


def _save_processed_data(images_train, labels_train,
                         images_val, labels_val,
                         images_test, labels_test):

    processed_data = {'images_train': images_train, 'labels_train': labels_train,
                      'images_val': images_val, 'labels_val': labels_val,
                      'images_test': images_test, 'labels_test': labels_test}

    repo_dir = os.path.dirname(__file__)
    save_path = os.path.join(
        repo_dir, 'data', 'brain_tumor_segmentation', 'processed_data.npz')

    np.savez(save_path, **processed_data)


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


def _get_images_masks(path: str, dataset: int):
    '''
    Get image and mask data for all tumor types. This processing step tosses
    information about tumor type, so the mask only contains labels for no tumor
    (0) and tumor (255).
    Arguments:
        path: Path to raw downloaded data.
        dataset: 0: none, 1: glioma, 2: meningioma, 3: pituitary
    Returns:
        images, masks: Numpy array (sample, row, column)
    '''

    # used to store data for all tumor types
    images = []
    image_files = []
    masks = []
    mask_files = []

    image_dir = os.path.join(
        path, 'Brain Tumor Segmentation Dataset', 'image', str(dataset))
    images_subset, image_files_subset = ip.get_images(
        image_dir)
    images += images_subset
    image_files += image_files_subset

    mask_dir = os.path.join(
        path, 'Brain Tumor Segmentation Dataset', 'mask', str(dataset))
    masks_subset, mask_files_subset = ip.get_images(mask_dir)
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
            images.pop(ii)
            image_files.pop(ii)

    # crawl masks in reverse order to make item removal easier
    for mm in np.flip(np.arange(len(mask_files))):
        mask_file = mask_files[mm]
        root, ext = os.path.splitext(mask_file)
        image_file = root[:-2] + ext  # remove '_m' from mask file name
        if image_file not in image_files:  # no corresponding image, so toss
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

    # input masks and resized masks are non-binary; convert values to 0 or 255
    masks = ip.threshold_masks(masks)

    return images, masks


if __name__ == '__main__':
    download_and_preprocess_data(output_size=256)
