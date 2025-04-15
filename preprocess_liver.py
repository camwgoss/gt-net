import os
from PIL import Image
import numpy as np

import utils.image_processing as ip
import utils.split_data as split_data


def download_and_preprocess_data(output_size: int = 256, augmentation=None,augmentation_sets: int = 1):
    '''
    Ensure you've downloaded the liver tumor segmentation dataset from
    https://www.kaggle.com/datasets/ag3ntsp1d3rx/litsdataset2/data?select=images
    as described in the readMe.
   
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
    # raw_path = os.path.join(repo_dir, 'data','liver_tumor_segmentation','trial','raw')
    raw_path = os.path.join(repo_dir, 'data','liver_tumor_segmentation','subset')
    datasets=[0,1]

    final_images_train = np.empty([0, output_size, output_size])
    final_masks_train = np.empty([0, output_size, output_size])
    final_images_val = np.empty([0, output_size, output_size])
    final_masks_val = np.empty([0, output_size, output_size])
    final_images_test = np.empty([0, output_size, output_size])
    final_masks_test = np.empty([0, output_size, output_size])
           
    for dataset in datasets:

        images_raw, masks_raw = _get_images_masks(raw_path,dataset)
        #raw plot
        fig = ip.plot_images_masks(images_raw, masks_raw)
        save_path = os.path.join(
            repo_dir, 'data', 'liver_tumor_segmentation','subset')
            # repo_dir, 'data', 'liver_tumor_segmentation','trial', 'raw')
        fig.savefig(save_path, bbox_inches='tight')
        
        #split
        images_train_raw, images_val_raw, images_test_raw = split_data.split_data(
            images_raw)
        masks_train_raw, masks_val_raw, masks_test_raw = split_data.split_data(
            masks_raw)
    
        #process raw images        
        images_train, masks_train = ip.resize_images(
            images_train_raw, masks_train_raw, output_size)
        images_val, masks_val = ip.resize_images(
            images_val_raw, masks_val_raw, output_size)
        images_test, masks_test = ip.resize_images(
            images_test_raw, masks_test_raw, output_size)
    
        # convert masks {0, 255} to labels {0, 1}
        labels_train = ip.masks_to_labels(
            masks_train, label=dataset)
        labels_val = ip.masks_to_labels(
            masks_val, label=dataset)
        labels_test = ip.masks_to_labels(
            masks_test, label=dataset)
    
    
        final_images_train = np.concatenate(
            [final_images_train, images_train], axis=0)
        final_masks_train = np.concatenate(
            [final_masks_train, labels_train], axis=0)
        final_images_val = np.concatenate(
            [final_images_val, images_val], axis=0)
        final_masks_val = np.concatenate(
            [final_masks_val, labels_val], axis=0)
        final_images_test = np.concatenate(
            [final_images_test, images_test], axis=0)
        final_masks_test = np.concatenate(
            [final_masks_test, labels_test], axis=0)

        if augmentation is not None:  # augment training dataset
            for aa in range(augmentation_sets):
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
    
                final_images_train = np.concatenate(
                    [final_images_train, images_train], axis=0)
                final_masks_train = np.concatenate(
                    [final_masks_train, labels_train], axis=0)

    
    _save_processed_data(final_images_train, final_masks_train,
                         final_images_val, final_masks_val,
                         final_images_test, final_masks_test)    


    #processed plot 
    fig = ip.plot_images_masks(final_images_train, final_masks_train)
    repo_dir = os.path.dirname(__file__)
    save_path = os.path.join(
        repo_dir, 'data', 'liver_tumor_segmentation','processed')
        # repo_dir, 'data', 'liver_tumor_segmentation','trial', 'processed')
    fig.savefig(save_path, bbox_inches='tight')


def load_processed_data():
    '''
    Returns:
        images_train, masks_train, images_eval, masks_eval, images_test, masks_test:
        List of Numpy arrays containing image and mask data, (row, column, [channel]).
    '''

    repo_dir = os.path.dirname(__file__)
    data_path = os.path.join(
        repo_dir, 'data', 'liver_tumor_segmentation','processed_data.npz')
        # repo_dir, 'data', 'liver_tumor_segmentation', 'trial','processed_data.npz')
    processed_data = np.load(data_path)
    return processed_data

def _save_processed_data(images_train, masks_train,
                         images_val, masks_val,
                         images_test, masks_test):

    processed_data = {'images_train': images_train, 'labels_train': masks_train,
                      'images_val': images_val, 'labels_val': masks_val,
                      'images_test': images_test, 'labels_test': masks_test}

    repo_dir = os.path.dirname(__file__)
    save_path = os.path.join(
        repo_dir, 'data', 'liver_tumor_segmentation','processed_data.npz')
        # repo_dir, 'data', 'liver_tumor_segmentation','trial', 'processed_data.npz')

    np.savez(save_path, **processed_data)
    
    
def _get_images_masks(path: str, dataset: int):
    '''
    Get image and mask data for all tumor types. This processing step tosses
    information about tumor type, so the mask only contains labels for no tumor
    (0) and tumor (255).
    Arguments:
        path: Path to raw downloaded data.
        dataset: 0: none, 1: tumor
    Returns:
        images, masks: Numpy array (sample, row, column)
    '''

    # used to store data for all tumor types
    images = []
    image_files = []
    masks = []
    mask_files = []


    image_dir = os.path.join(path,'image', str(dataset))

    images_subset, image_files_subset = ip.get_images(
        image_dir)
    images += images_subset
    image_files += image_files_subset

    mask_dir = os.path.join(path, 'mask', str(dataset))
    masks_subset, mask_files_subset = ip.get_images(mask_dir)
    masks += masks_subset
    mask_files += mask_files_subset


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
    # masks = image_processing.threshold_masks(masks)
    masks = ip.liver_threshold_masks(masks)

    return images, masks


if __name__ == '__main__':
    download_and_preprocess_data(output_size=256)
