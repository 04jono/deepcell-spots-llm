import logging
import numpy as np


def min_max_normalize(image, clip=False):
    """Normalize image data by subtracting minimum pixel value and
     dividing by the maximum pixel value.

    Args:
        image (numpy.array): 4D numpy array of image data.
        clip (boolean): Defaults to false. Determines if pixel
            values are clipped by percentile.

    Returns:
        numpy.array: normalized image data.
    """
    if not np.issubdtype(image.dtype, np.floating):
        logging.info('Converting image dtype to float')
    image = image.astype('float32')

    if not len(np.shape(image)) == 4:
        raise ValueError('Image must be 4D, input image shape was'
                         ' {}.'.format(np.shape(image)))

    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            img = image[batch, ..., channel]

            if clip:
                img = np.clip(img, a_min=np.percentile(img, 0.01), a_max=np.percentile(img, 99.9))

            min_val = np.min(img)
            max_val = np.max(img)
            normal_image = (img - min_val) / (max_val - min_val)

            image[batch, ..., channel] = normal_image
    return image

def gaussian_filter(image, clip=False):
    """Apply a gaussian filter to the image.

    Args:
        image (numpy.array): 4D numpy array of image data.

    Returns:
        numpy.array: image data with gaussian filter applied.
    """
    from scipy.ndimage import gaussian_filter as gf

    if not np.issubdtype(image.dtype, np.floating):
        logging.info('Converting image dtype to float')
    image = image.astype('float32')

    if not len(np.shape(image)) == 4:
        raise ValueError('Image must be 4D, input image shape was'
                         ' {}.'.format(np.shape(image)))

    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):

            img = image[batch, ..., channel]
            if clip:
                img = np.clip(img, a_min=np.percentile(img, 0.01), a_max=np.percentile(img, 99.9))

            image[batch, ..., channel] = gf(img, sigma=1)

    return image

def contrast_stretch(image, clip=False):
    """Apply a simple contrast stretch to the image data.

    Args:
        image (numpy.array): 4D numpy array of image data.
        clip (boolean): Defaults to false. Determines if pixel
            values are clipped by percentile.

    Returns:
        numpy.array: normalized image data with contrast stretch applied.
    """
    if not np.issubdtype(image.dtype, np.floating):
        logging.info('Converting image dtype to float')
    image = image.astype('float32')

    if not len(np.shape(image)) == 4:
        raise ValueError('Image must be 4D, input image shape was'
                        '{}.'.format(np.shape(image)))

    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            img = image[batch,..., channel]

            if clip:
                img = np.clip(img, a_min=np.percentile(img, 0.01), a_max=np.percentile(img, 99.9))

            contrast_stretch_value = img / (img.max() + 1)
            contrast_stretched_img = contrast_stretch_value * (img.max() + 1)

            image[batch,..., channel] = contrast_stretched_img
    return image