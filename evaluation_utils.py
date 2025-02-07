import os
import imageio
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

from tensorflow.keras.utils import get_file
from tensorflow.keras.utils import to_categorical

from deepcell.utils.plot_utils import create_rgb_image
from deepcell.datasets import SpotNetExampleData, SpotNet
from deepcell_spots.applications import SpotDetection
from deepcell_spots.dotnet_losses import DotNetLosses

from deepcell_spots.utils.augmentation_utils import subpixel_distance_transform

from deepcell_spots.image_generators import ImageFullyConvDotIterator

from dotenv import load_dotenv

import inspect

load_dotenv()

def write_results(preprocessing_fn, class_loss: int, regress_loss: int):
    '''
    Write the results of evaluation to the function bank JSON.
    
    Requires:
    preprocessing_fn, the function
    class_loss, the classification loss
    regress_loss, the regression_loss
    
    '''
    
    with open('function_bank.json', 'r') as file:
        json_array = json.load(file)

    with open('function_bank.json', 'w') as file:
        json_data = {
            "code": inspect.getsource(preprocessing_fn),
            "class_loss": class_loss,
            "regress_loss": regress_loss
        }
        json_array.append(json_data)
        json.dump(json_array, file)

def evaluate_spots(preprocessing_fn):
    '''Evaluate the classification and regression loss over the test set.
    
    Requires: preprocessing_fn, a function of type (image, bool) -> image
    
    Returns: classification loss average, regression loss average'''
    
    spots_data = np.load('spot_data/SpotNet-v1_1/val.npz', allow_pickle=True)
    spots_images = spots_data['X']
    spots_truth = spots_data['y']

    app = SpotDetection(preprocessing_fn=preprocessing_fn)


    pred, class_output, regress_output = app.predict(spots_images, batch_size=spots_images.shape[0], threshold=0.95)

    def point_list_to_annotations(points, image_shape, dy=1, dx=1):
            """ Generate label images used in loss calculation from point labels.

            Args:
                points (np.array): array of size (N, 2) which contains points in the format [y, x].
                image_shape (tuple): shape of 2-dimensional image.
                dy: pixel y width.
                dx: pixel x width.

            Returns:
                annotations (dict): Dictionary with two keys, `detections` and `offset`.
                    - `detections` is array of shape (image_shape,2) with pixels one hot encoding
                    spot locations.
                    - `offset` is array of shape (image_shape,2) with pixel values equal to
                    signed distance to nearest spot in x- and y-directions.
            """

            contains_point = np.zeros(image_shape)
            for ind, [y, x] in enumerate(points):
                nearest_pixel_x_ind = int(round(x / dx))
                nearest_pixel_y_ind = int(round(y / dy))
                contains_point[nearest_pixel_y_ind, nearest_pixel_x_ind] = 1

            delta_y, delta_x, _ = subpixel_distance_transform(
                points, image_shape, dy=1, dx=1)
            offset = np.stack((delta_y, delta_x), axis=-1)

            one_hot_encoded_cp = to_categorical(contains_point)

            annotations = {'detections': one_hot_encoded_cp, 'offset': offset}
            return annotations



    losses = DotNetLosses()

    sum_classification_loss = 0
    sum_regression_loss = 0
    for i in range(class_output.shape[0]):
        spot_annotations = point_list_to_annotations(spots_truth[i], spots_images.shape[1:3])
        sum_classification_loss += losses.classification_loss(spot_annotations['detections'], class_output[i]).numpy()
        sum_regression_loss += losses.regression_loss(spot_annotations['offset'], regress_output[i]).numpy()
    
    return sum_classification_loss / class_output.shape[0], sum_regression_loss / regress_output.shape[0]

def hyperparameter_search(preprocessing_fn):
    '''Perform a hyperparameter search over the preprocessing function.
    
    Requires: preprocessing_fn, a function of type (image, bool, ...) -> image, where ... can contain any number of hyperparameters with default values.
    
    Returns: the best hyperparameters'''

    
    pass