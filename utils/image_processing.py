import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def get_images(image_dir: str, grayscale=True):
    '''
    Get all the images in a directory. Images must have the same dimensions.
    Arguments:
        image_dir: Directory containing images.
        grayscale: whether to return images in grayscale.
    Returns:
        images: List of Numpy arrays, (row, column).
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


def threshold_masks(masks):
    '''
    When masks are stored using lossy compression (e.g., JPEG), they take on
    values other than 0 and 255. Use thresholding to convert to binary masks.
    This operation is performed in place.
    Arguments:
        masks: List of Numpy arrays, (row, column).
    Returns:
        masks: Thresholded masks, (row, column).
    '''

    for mask in masks:
        mask[mask < 128] = 0
        mask[mask >= 128] = 255
    return masks


def masks_to_labels(masks, label: int):
    '''
    Convert binary masks with values 0 or 255 to labels with values 0 or label.
    Arguments:
        masks: List of Numpy arrays, (row, column). Values 0 or 255.
        label: Label to replace non-zero values with.
    Returns:
        labels: List of Numpy arrays, (row, column)
    '''

    masks = np.copy(masks)
    for mask in masks:
        mask[mask != 0] = label
    return masks


def crop_images(images: list, masks: list = None, output_size: int = 256):
    '''
    Randomly extract a square section from each image. If the image has a 
    dimension smaller than section_size, that image will be removed.
    Arguments:
        images: List of Numpy arrays, (row, column).
        masks: Must have identical dimensions to images.
        output_size: Dimension in pixels of square section to extract.
    Returns:
        images_out: List of Numpy arrays containing image sections.
        masks_out: This will only be returned if masks were provided.
    '''

    images_out = []
    masks_out = []

    for ii in range(len(images)):

        image = images[ii]
        rows = image.shape[0]
        cols = image.shape[1]

        if output_size > rows or output_size > cols:
            pass  # the section is larger than the image, so toss
        else:
            row_start = np.random.randint(rows - output_size + 1)
            row_end = row_start + output_size

            col_start = np.random.randint(cols - output_size + 1)
            col_end = col_start + output_size

            image_out = image[row_start:row_end, col_start:col_end]
            images_out.append(image_out)

            if masks is not None:
                mask = masks[ii]
                mask_out = mask[row_start:row_end, col_start:col_end]
                masks_out.append(mask_out)

    images_out = np.array(images_out)
    if masks is None:
        return images_out
    else:
        masks_out = np.array(masks_out)
        return images_out, masks_out


def resize_images(images: list, masks: list = None, output_size: int = 256):
    '''
    Resize images to a square with the specified resolution. Perform a centered
    square crop prior to resizing to prevent stretching.
    Arguments:
        images: List of Numpy arrays, (row, column, [channel]).
        masks: Must have identical dimensions to images.
        output_size: Dimension in pixels of square section to extract.
    Returns:
        image_sections: List of Numpy arrays containing image sections.
        mask_sections: This will only be returned if masks were provided.
    '''

    images_out = []
    masks_out = []

    for ii in range(len(images)):

        image = images[ii]
        rows = image.shape[0]
        cols = image.shape[1]

        # crop to largest centered square possible
        row_start = 0
        row_end = -1
        col_start = 0
        col_end = -1
        if rows > cols:
            row_start = int((rows - cols)/2)
            row_end = row_start + cols
        elif cols > rows:
            col_start = int((cols - rows)/2)
            col_end = col_start + rows
        image_out = image[row_start:row_end, col_start:col_end]

        # resize
        image_out = Image.fromarray(image_out)
        image_out = image_out.resize([output_size, output_size])
        image_out = np.array(image_out)

        images_out.append(image_out)

        if masks is not None:  # same processing as image
            mask = masks[ii]

            mask_out = mask[row_start:row_end, col_start:col_end]  # crop

            # resize
            mask_out = Image.fromarray(mask_out)
            mask_out = mask_out.resize([output_size, output_size])
            mask_out = np.array(mask_out)

            masks_out.append(mask_out)

    images_out = np.array(images_out)
    if masks is None:
        return images_out
    else:
        # resampling converted binary mask to non-binary mask; convert back
        mask_out = threshold_masks(masks_out)
        masks_out = np.array(masks_out)
        return images_out, masks_out


def plot_images_masks(images: list, masks: list, num_samples=10):
    '''
    Plot images and masks side by side.
    Arguments:
        images, masks: List of Numpy arrays, (row, column, [channel]).
        num_samples: Number of samples to plot
    Returns:
        fig: Figure containing images.
    '''

    if num_samples > len(images):  # not enough samples to plot, so plot fewer
        num_samples = len(images)

    # randomly select which images/masks to plot
    samples = np.random.choice(len(images), num_samples, replace=False)
    for sample in samples:
        images_subset = [images[ss] for ss in samples]
        masks_subset = [masks[ss] for ss in samples]

    fig, axes = plt.subplots(num_samples, 2, dpi=200, figsize=[2, num_samples])

    for ii in range(len(images_subset)):
        axes[ii, 0].imshow(images_subset[ii], cmap='plasma')
        axes[ii, 1].imshow(masks_subset[ii], cmap='plasma')
        axes[ii, 0].axis('off')
        axes[ii, 1].axis('off')

    return fig
