import matplotlib.pyplot as plt
#import nibabel as nib
import os
import utils.image_processing as ip
import utils.split_data as split_data
import numpy as np
import matplotlib
import pickle
import gc

def download_and_preprocess_data(output_size: int = 256, dataset: str='rel_25percent.pkl',
                                 save_name: str='processed_data',
                                 augmentation=None,augmentation_sets: int = 1):
    repo_dir = os.path.dirname(__file__)
    
    with open(dataset,'rb') as f:
        images_raw = pickle.load(f)
        masks_raw = pickle.load(f)

    final_images_train = np.empty([0, output_size, output_size])
    final_masks_train = np.empty([0, output_size, output_size])
    final_images_val = np.empty([0, output_size, output_size])
    final_masks_val = np.empty([0, output_size, output_size])
    final_images_test = np.empty([0, output_size, output_size])
    final_masks_test = np.empty([0, output_size, output_size])


    for m in masks_raw:
        m[m==1]=0
        m[m==2]=255

    #split
    images_train_raw, images_val_raw, images_test_raw = split_data.split_data(
        images_raw)
    masks_train_raw, masks_val_raw, masks_test_raw = split_data.split_data(
        masks_raw)

    #process raw images        
    images_train, labels_train = ip.resize_images(
        images_train_raw, masks_train_raw, output_size)
    images_val, labels_val = ip.resize_images(
        images_val_raw, masks_val_raw, output_size)
    images_test, labels_test = ip.resize_images(
        images_test_raw, masks_test_raw, output_size)

    # convert masks {0, 255} to labels {0, 1} #not tumor, tumor
    labels_train = ip.liver_masks_to_labels(
        labels_train)
    labels_val = ip.liver_masks_to_labels(
        labels_val)
    labels_test = ip.liver_masks_to_labels(
        labels_test)


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
            elif augmentation == 'blur':
                    processor = ip.blur_images
            elif augmentation == 'elastic_deformation':
                processor = ip.elastically_deform_images
            else:
                raise Exception('Error: Unknown augmentation provided.')

            images_train, masks_train = processor(
                images_train_raw, masks_train_raw)
            labels_train = ip.liver_masks_to_labels(masks_train)

            final_images_train = np.concatenate(
                [final_images_train, images_train], axis=0)
            final_masks_train = np.concatenate(
                [final_masks_train, labels_train], axis=0)
            
    _save_processed_data(final_images_train, final_masks_train,
                          final_images_val, final_masks_val,
                          final_images_test, final_masks_test,save_name)    

            
    fig = ip.plot_images_labels(final_images_train, final_masks_train)
    save_path = os.path.join(
        repo_dir, 'data','liver_tumor_segmentation',save_name+'.png')
    fig.savefig(save_path, bbox_inches='tight')


def load_processed_data(filename: str='processed_data.npz'):
    '''
    Returns:
        images_train, masks_train, images_eval, masks_eval, images_test, masks_test:
        List of Numpy arrays containing image and mask data, (row, column, [channel]).
    '''

    repo_dir = os.path.dirname(__file__)
    data_path = os.path.join(
        repo_dir, 'data', 'liver_tumor_segmentation',filename)
    processed_data = np.load(data_path)
    return processed_data

def _save_processed_data(images_train, masks_train,
                         images_val, masks_val,
                         images_test, masks_test,
                         save_name:str='processed_data'):

    processed_data = {'images_train': images_train, 'labels_train': masks_train,
                      'images_val': images_val, 'labels_val': masks_val,
                      'images_test': images_test, 'labels_test': masks_test}

    repo_dir = os.path.dirname(__file__)
    save_path = os.path.join(
        repo_dir, 'data', save_name+'.npz')
    np.savez(save_path, **processed_data)


if __name__ == '__main__':
    download_and_preprocess_data(output_size=125,dataset='full_10percent.pkl',
                                 save_name='processed_fullBin125')
        
