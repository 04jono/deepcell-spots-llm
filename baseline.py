import cv2 as cv
import numpy as np
from itertools import combinations, product
from evaluation_utils import evaluate_spots
import csv

from dotenv import load_dotenv

load_dotenv()

import time

start_time = time.time()

def create_filter_pipeline(func1_name, params1, func2_name, params2, filter_functions):
    """
    Returns a callable that applies two filters sequentially with specified parameters.
    """
    def pipeline(img, clip = False):
        if len(img.shape) != 4:
            raise ValueError('Image must be 4D, input image shape was {}.'.format(img.shape))

        for batch in range(img.shape[0]):
            for channel in range(img.shape[-1]):
                temp_img = filter_functions[func1_name](img[batch, ..., channel], *(params1 if isinstance(params1, tuple) else (params1,))) if params1 else filter_functions[func1_name](img[batch, ..., channel])
                result_img = filter_functions[func2_name](temp_img, *(params2 if isinstance(params2, tuple) else (params2,))) if params2 else filter_functions[func2_name](temp_img)
                img[batch, ..., channel] = result_img
        return img

    return pipeline

def apply_filters():

    # Functions
    filter_functions = {
        'bilateralFilter': lambda img, d, sigmaColor, sigmaSpace: cv.bilateralFilter(img, d, sigmaColor, sigmaSpace),
        'blur': lambda img, ksize: cv.blur(img, (ksize, ksize)),
        'boxFilter': lambda img, ksize: cv.boxFilter(img, -1, (ksize, ksize)),
        'dilate': lambda img, ksize: cv.dilate(img, cv.getStructuringElement(cv.MORPH_RECT, (ksize, ksize))),
        'erode': lambda img, ksize: cv.erode(img, cv.getStructuringElement(cv.MORPH_RECT, (ksize, ksize))),
        'filter2D': lambda img, ksize: cv.filter2D(img, -1, np.ones((ksize, ksize), np.float32) / (ksize * ksize)),
        'GaussianBlur': lambda img, ksize: cv.GaussianBlur(img, (ksize, ksize), 0),
        'Laplacian': lambda img, ksize: cv.Laplacian(img, cv.CV_64F, ksize=ksize),
        'medianBlur': lambda img, ksize: cv.medianBlur(img, ksize),
        'pyrDown': lambda img: cv.pyrDown(img),
        'pyrMeanShiftFiltering': lambda img, sp, sr: cv.pyrMeanShiftFiltering(img, sp, sr),
        'pyrUp': lambda img: cv.pyrUp(img),
        'Scharr': lambda img: cv.Scharr(img, cv.CV_64F, 1, 0),
        'sepFilter2D': lambda img: cv.sepFilter2D(img, -1, np.array([1, 4, 6, 4, 1]), np.array([1, 4, 6, 4, 1])),
        'Sobel': lambda img, ksize: cv.Sobel(img, cv.CV_64F, 1, 0, ksize=ksize),
        'spatialGradient': lambda img: cv.spatialGradient(img)[0],
        'sqrBoxFilter': lambda img, ksize: cv.sqrBoxFilter(img, -1, (ksize, ksize)),
        'stackBlur': lambda img, ksize: cv.blur(img, (ksize, ksize))  # Placeholder for stackBlur if not available
    }

    # Hyperparameters
    param_ranges = {
        'bilateralFilter': [(5, 75, 75), (9, 100, 100)],
        'blur': [3, 5, 7],
        'boxFilter': [3, 5, 7],
        'dilate': [3, 5, 7],
        'erode': [3, 5, 7],
        'filter2D': [3, 5, 7],
        'GaussianBlur': [3, 5, 7],
        'Laplacian': [1, 3, 5],
        'medianBlur': [3, 5, 7],
        'pyrMeanShiftFiltering': [(10, 20), (20, 40)],
        'Sobel': [1, 3, 5],
        'sqrBoxFilter': [3, 5, 7],
        'stackBlur': [3, 5, 7]
    }

    combinations_of_filters = list(combinations(filter_functions.keys(), 2))

    output_file = 'results.csv'

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Function1', 'Params1', 'Function2', 'Params2', 'Class Loss', 'Regress Loss'])

        for func1_name, func2_name in combinations_of_filters:
            params1 = param_ranges.get(func1_name, [()])
            params2 = param_ranges.get(func2_name, [()])

            for p1, p2 in product(params1, params2):
                
                try:
                    pipeline = create_filter_pipeline(func1_name, p1, func2_name, p2, filter_functions)
                    class_loss, regress_loss = evaluate_spots(pipeline)
                
                    writer.writerow([func1_name, p1, func2_name, p2, class_loss, regress_loss])
                except Exception as e:
                    writer.writerow([func1_name, p1, func2_name, p2, 0, str(e)])


apply_filters()

end_time = time.time()

execution_time = end_time - start_time

print("Execution time:", execution_time, "seconds")
