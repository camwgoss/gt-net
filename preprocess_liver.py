import os
from PIL import Image
import numpy as np

import utils.image_processing as image_processing
import utils.split_data as split_data


def download_and_preprocess_data(output_size: int = 256, output_type='crop'):
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
    
    if output_type == 'crop':
        processor = image_processing.crop_images
    elif output_type == 'resize':
        processor = image_processing.resize_images
        
    for dataset in datasets:

        images_raw, masks_raw = _get_images_masks(raw_path,dataset)
        #raw plot
        fig = image_processing.plot_images_masks(images_raw, masks_raw)
        save_path = os.path.join(
            repo_dir, 'data', 'liver_tumor_segmentation','subset')
            # repo_dir, 'data', 'liver_tumor_segmentation','trial', 'raw')
        fig.savefig(save_path, bbox_inches='tight')
        
        #split
        images_train, images_val, images_test = split_data.split_data(images_raw)
        masks_train, masks_val, masks_test = split_data.split_data(masks_raw)
    
        # process raw images
        images_train, masks_train = processor(
            images_train, masks_train, output_size)
        images_val, masks_val = processor(
            images_val, masks_val, output_size)
        images_test, masks_test = processor(
            images_test, masks_test, output_size)
    
        # convert masks {0, 255} to labels {0, 1}
        labels_train = image_processing.masks_to_labels(
            masks_train, label=dataset)
        labels_val = image_processing.masks_to_labels(
            masks_val, label=dataset)
        labels_test = image_processing.masks_to_labels(
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

    
    _save_processed_data(final_images_train, final_masks_train,
                         final_images_val, final_masks_val,
                         final_images_test, final_masks_test)    


    #processed plot 
    fig = image_processing.plot_images_masks(images_train, masks_train)
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

    images_subset, image_files_subset = image_processing.get_images(
        image_dir)
    images += images_subset
    image_files += image_files_subset

    mask_dir = os.path.join(path, 'mask', str(dataset))
    masks_subset, mask_files_subset = image_processing.get_images(mask_dir)
    masks += masks_subset
    mask_files += mask_files_subset

    # # This is not a perfect dataset, so there are a couple images without
    # # corresponding masks or vice versa. Crawl through masks and images and
    # # toss un-paired entries.

    # # crawl images in reverse order to make item removal easier
    # for ii in np.flip(np.arange(len(image_files))):
    #     image_file = image_files[ii]
    #     root, ext = os.path.splitext(image_file)
    #     mask_file = root + '_m' + ext  # mask files add '_m' to file name
    #     if mask_file not in mask_files:  # no corresponding mask
    #         images.pop(ii)
    #         image_files.pop(ii)

    # # crawl masks in reverse order to make item removal easier
    # for mm in np.flip(np.arange(len(mask_files))):
    #     mask_file = mask_files[mm]
    #     root, ext = os.path.splitext(mask_file)
    #     image_file = root[:-2] + ext  # remove '_m' from mask file name
    #     if image_file not in image_files:  # no corresponding image, so toss
    #         masks.pop(mm)
    #         mask_files.pop(mm)

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
    masks = image_processing.liver_threshold_masks(masks)

    return images, masks


if __name__ == '__main__':
    download_and_preprocess_data(output_size=256, output_type='resize')


