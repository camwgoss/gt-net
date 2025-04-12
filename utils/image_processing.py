import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch


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
        images_out: Numpy array (sample, row, col).
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
        images: List of Numpy arrays, (row, column).
        masks: Must have identical dimensions to images.
        output_size: Dimension in pixels of square section to extract.
    Returns:
        images_out: Numpy array (sample, row, col).
        masks_out: This will only be returned if masks were provided.
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


def rotate_images(images: list, masks: list = None, output_size: int = 256):
    '''
    Randomly rotate images and then resize them to a square with the specified 
    resolution. Random rotation angle standard deviation is 10 deg.
    Arguments:
        images: List of Numpy arrays, (row, column).
        masks: Must have identical dimensions to images.
        output_size: Dimension in pixels of square section to extract.
    Returns:
        image_sections: Numpy array (sample, row, col).
        mask_sections: This will only be returned if masks were provided.
    '''

    images_out = []
    masks_out = []

    for ii in range(len(images)):
        rotation_angle = np.random.normal(scale=10)

        image = images[ii]
        image = Image.fromarray(image)
        image = image.rotate(rotation_angle)
        image = np.array(image)
        images_out.append(image)

        if masks is not None:  # same processing as image
            mask = masks[ii]
            mask = Image.fromarray(mask)
            mask = mask.rotate(rotation_angle)
            mask = np.array(mask)
            masks_out.append(mask)

    if masks is None:
        images_out = resize_images(images_out, output_size=output_size)
        images_out = np.array(images_out)
        return images_out
    else:
        images_out, masks_out = resize_images(
            images_out, masks_out, output_size=output_size)
        images_out = np.array(images_out)
        masks_out = np.array(masks_out)
        return images_out, masks_out


def elastically_deform_images(images: list, masks: list = None, output_size: int = 256):
    '''
    Randomly elastically deform images using bi-cubic interpolation. A cubic
    function is fit to each row/column of a 4x4 grid of control points, where
    each point's row and column pixel location is randomly shifted according to
    a normal distribution with a 10 pixel standard deviation. A similar
    approach and 10 pixel std was used in the original U-Net paper.
    Arguments:
        images: List of Numpy arrays, (row, column).
        masks: Must have identical dimensions to images.
        output_size: Dimension in pixels of square section to extract.
    Returns:
        image_sections: Numpy array (sample, row, col).
        mask_sections: This will only be returned if masks were provided.
    '''

    images_out = []
    masks_out = []

    for ii in range(len(images)):
        image = images[ii]

        rows = image.shape[0]
        cols = image.shape[1]

        # X @ coefficients = deformations, where X is the polynomial matrix,
        # coefficients is the coefficients for the row/col deformation,
        # and deformations is the row/col deformations

        x = np.arange(0, rows, (rows-1)/3)
        x = np.repeat(x, 4)  # [1, 2] -> [1, 1, 2, 2]

        y = np.arange(0, cols, (cols-1)/3)
        y = np.tile(y, 4)  # [1, 2] -> [1, 2, 1, 2]

        X = np.array([x**0*y**0, x**0*y**1, x**0*y**2, x**0*y**3,
                      x**1*y**0, x**1*y**1, x**1*y**2, x**1*y**3,
                      x**2*y**0, x**2*y**1, x**2*y**2, x**2*y**3,
                      x**3*y**0, x**3*y**1, x**3*y**2, x**3*y**3]).T

        std = 10  # pixels
        deformations_row = std * np.random.randn(16)
        deformations_col = std * np.random.randn(16)

        # solve the linear system
        coefficients_row = np.linalg.solve(X, deformations_row)
        coefficients_col = np.linalg.solve(X, deformations_col)

        # reshape coefficients for use with polyval2d
        coefficients_row = coefficients_row.reshape([4, 4])
        coefficients_col = coefficients_col.reshape([4, 4])

        # all pixels
        x_grid = np.arange(0, rows)
        x_grid = np.repeat(x_grid, cols)  # [1, 2] -> [1, 1, 2, 2]
        x_grid = np.reshape(x_grid, [rows, cols])

        y_grid = np.arange(0, cols)
        y_grid = np.tile(y_grid, rows)  # [1, 2] -> [1, 2, 1, 2]
        y_grid = np.reshape(y_grid, [rows, cols])

        # how much to shift each pixel
        x_shift = np.polynomial.polynomial.polyval2d(
            x_grid, y_grid, coefficients_row)
        y_shift = np.polynomial.polynomial.polyval2d(
            x_grid, y_grid, coefficients_col)

        # some of these indices will be < 0 or > image rows/cols
        x_shifted_raw = (x_grid + x_shift).astype(int)
        y_shifted_raw = (y_grid + y_shift).astype(int)

        # this is the actual amount to circle shift each row/col
        x_shifted = x_shifted_raw % rows
        y_shifted = y_shifted_raw % cols

        image = image[x_shifted, y_grid]  # elastically deform rows/cols
        image[x_shifted != x_shifted_raw] = 0  # circle-shifted data black
        image = image[x_grid, y_shifted]
        image[y_shifted != y_shifted_raw] = 0

        images_out.append(image)

        if masks is not None:  # same processing as image
            mask = masks[ii]

            mask = mask[x_shifted, y_grid]  # elastically deform rows/cols
            mask[x_shifted != x_shifted_raw] = 0  # circle-shifted data black
            mask = mask[x_grid, y_shifted]
            mask[y_shifted != y_shifted_raw] = 0

            masks_out.append(mask)

    if masks is None:
        images_out = resize_images(images_out, output_size=output_size)
        images_out = np.array(images_out)
        return images_out
    else:
        images_out, masks_out = resize_images(
            images_out, masks_out, output_size=output_size)
        images_out = np.array(images_out)
        masks_out = np.array(masks_out)
        return images_out, masks_out


def plot_images_labels(images: np.array, labels: np.array,
                       labels_predicted: np.array = None, num_samples=10):
    '''
    Plot images and labels side by side. Can optionally plot predicted labels.
    Arguments:
        images, labels, labels_predicted: Numpy array (sample, row, column) or
        list of Numpy arrays (row, column)
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
        labels_subset = [labels[ss] for ss in samples]

        if labels_predicted is not None:
            labels_predicted_subset = [labels_predicted[ss] for ss in samples]

    # make figure
    if labels_predicted is not None:
        columns = 3
    else:
        columns = 2
    fig, axes = plt.subplots(num_samples, columns,
                             dpi=200, figsize=[columns, num_samples])

    # plot data
    if type(labels) is np.ndarray or type(labels) is torch.Tensor:
        vmax = labels.max()  # np.array (sample, row, col)
    else:
        vmax = None  # list of np.array (row, col)

    for ii in range(len(images_subset)):
        axes[ii, 0].imshow(images_subset[ii], cmap='inferno')
        axes[ii, 1].imshow(labels_subset[ii],
                           cmap='inferno', vmax=vmax)
        axes[ii, 0].axis('off')
        axes[ii, 1].axis('off')

        if labels_predicted is not None:
            axes[ii, 2].imshow(labels_predicted_subset[ii],
                               cmap='inferno', vmax=vmax)
            axes[ii, 2].axis('off')

    # set titles
    axes[0, 0].set_title('Image')
    axes[0, 1].set_title('Label')
    if labels_predicted is not None:
        axes[0, 2].set_title('Predict')

    return fig
