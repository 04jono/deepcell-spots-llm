def prepare_modification_message(context: str):
    task_message = """
    
    In a previous iteration of this task, the pipeline created:
    
    """ + context + """
    
    Your tasks are:
    
    1. Read in ./function_bank.json, which has evaluation results of previous iterations of this task. Based on the loss, create a small modification of one of the best performing functions, i.e. adjust a hyperparameter. This new takes an image as  input and returns a transformed/preprocessed image for the downstream pretrained spot detection model. ./example_bank.py has examples of functions that will run successfully, consider the syntax of these examples when debugging.

    2. Figure out how to evaluate the preprocessing function using evaluate_spots in ./evaluation_utils.py.

    3. After evaluation, write the loss results using write_results in ./evaluation_utils.py. 

    IMPORTANT:

    You will write a function in Python that transforms an image such that it will
    be easier to detect spots in it. You will be prompted with example functions.
    Do not write the same type of math function as the example functions.
    Only return the Python function itself. The function must have
    two arguments: image (numpy.array) and clip (boolean) and returns image (numpy.array). 
    The output image must be the same size as the input image.

    IMPORTANT:

    Here are OpenCV image processing functions you can use, or to inspire function compositions:
                        
    cv.bilateralFilter(src, d, sigmaColor, sigmaSpace[, dst[, borderType]]) ->dst
    Applies the bilateral filter to an image.

    cv.blur(src, ksize[, dst[, anchor[, borderType]]]) ->dst
    Blurs an image using the normalized box filter.

    cv.boxFilter(src, ddepth, ksize[, dst[, anchor[, normalize[, borderType]]]]) ->dst
    Blurs an image using the box filter.

    cv.dilate(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) ->dst
    Dilates an image by using a specific structuring element.

    cv.erode(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) ->dst
    Erodes an image by using a specific structuring element.

    cv.filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]]) ->dst
    Convolves an image with the kernel.

    cv.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType[, hint]]]]) ->dst
    Blurs an image using a Gaussian filter.

    cv.getDerivKernels(dx, dy, ksize[, kx[, ky[, normalize[, ktype]]]]) ->kx, ky
    Returns filter coefficients for computing spatial image derivatives.

    cv.getGaborKernel(ksize, sigma, theta, lambd, gamma[, psi[, ktype]]) ->retval
    Returns Gabor filter coefficients.

    cv.getGaussianKernel(ksize, sigma[, ktype]) ->retval
    Returns Gaussian filter coefficients.

    cv.getStructuringElement(shape, ksize[, anchor]) ->retval
    Returns a structuring element of the specified size and shape for morphological operations.

    cv.Laplacian(src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]]) ->dst
    Calculates the Laplacian of an image.

    cv.medianBlur(src, ksize[, dst]) ->dst
    Blurs an image using the median filter.

    cv.pyrDown(src[, dst[, dstsize[, borderType]]]) ->dst
    Blurs an image and downsamples it.

    cv.pyrMeanShiftFiltering(src, sp, sr[, dst[, maxLevel[, termcrit]]]) ->dst
    Performs initial step of meanshift segmentation of an image.

    cv.pyrUp(src[, dst[, dstsize[, borderType]]]) ->dst
    Upsamples an image and then blurs it.

    cv.Scharr(src, ddepth, dx, dy[, dst[, scale[, delta[, borderType]]]]) ->dst
    Calculates the first x- or y- image derivative using Scharr operator.

    cv.sepFilter2D(src, ddepth, kernelX, kernelY[, dst[, anchor[, delta[, borderType]]]]) ->dst
    Applies a separable linear filter to an image.

    cv.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]) ->dst
    Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator.

    cv.spatialGradient(src[, dx[, dy[, ksize[, borderType]]]]) ->dx, dy
    Calculates the first order image derivative in both x and y using a Sobel operator.

    cv.sqrBoxFilter(src, ddepth, ksize[, dst[, anchor[, normalize[, borderType]]]]) ->dst
    Calculates the normalized sum of squares of the pixel values overlapping the filter.

    cv.stackBlur(src, ksize[, dst]) ->dst
    Blurs an image using the stackBlur.


    The environment is set up with the necessary libraries and the checkpoints are downloaded (if there are any).
    """
    return task_message


def prepare_exploration_message(context: str):
    task_message = """
    
    In a previous iteration of this task, the pipeline created:
    
    """ + context + """
    
    Your tasks are:
    
    1. View ./function_bank.json, which has evaluation results of previous iterations of this task.

    2. Create a function that takes an image as input and returns a transformed/preprocessed image for the downstream pretrained spot detection model. ./example_bank.py has examples of functions that will run successfully, consider the syntax of these examples when debugging.

    3. Figure out how to evaluate the preprocessing function using evaluate_spots in ./evaluation_utils.py.

    4. After evaluation, write the loss results using write_results in ./evaluation_utils.py. 

    IMPORTANT:

    You will write a function in Python that transforms an image such that it will
    be easier to detect spots in it. You will be prompted with example functions.
    Do not write the same type of math function as the example functions.
    Only return the Python function itself. The function must have
    two arguments: image (numpy.array) and clip (boolean) and returns image (numpy.array). 
    The output image must be the same size as the input image.

    IMPORTANT:

    Here are OpenCV image processing functions you can use, or to inspire function compositions:
                        
    cv.bilateralFilter(src, d, sigmaColor, sigmaSpace[, dst[, borderType]]) ->dst
    Applies the bilateral filter to an image.

    cv.blur(src, ksize[, dst[, anchor[, borderType]]]) ->dst
    Blurs an image using the normalized box filter.

    cv.boxFilter(src, ddepth, ksize[, dst[, anchor[, normalize[, borderType]]]]) ->dst
    Blurs an image using the box filter.

    cv.dilate(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) ->dst
    Dilates an image by using a specific structuring element.

    cv.erode(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) ->dst
    Erodes an image by using a specific structuring element.

    cv.filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]]) ->dst
    Convolves an image with the kernel.

    cv.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType[, hint]]]]) ->dst
    Blurs an image using a Gaussian filter.

    cv.getDerivKernels(dx, dy, ksize[, kx[, ky[, normalize[, ktype]]]]) ->kx, ky
    Returns filter coefficients for computing spatial image derivatives.

    cv.getGaborKernel(ksize, sigma, theta, lambd, gamma[, psi[, ktype]]) ->retval
    Returns Gabor filter coefficients.

    cv.getGaussianKernel(ksize, sigma[, ktype]) ->retval
    Returns Gaussian filter coefficients.

    cv.getStructuringElement(shape, ksize[, anchor]) ->retval
    Returns a structuring element of the specified size and shape for morphological operations.

    cv.Laplacian(src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]]) ->dst
    Calculates the Laplacian of an image.

    cv.medianBlur(src, ksize[, dst]) ->dst
    Blurs an image using the median filter.

    cv.pyrDown(src[, dst[, dstsize[, borderType]]]) ->dst
    Blurs an image and downsamples it.

    cv.pyrMeanShiftFiltering(src, sp, sr[, dst[, maxLevel[, termcrit]]]) ->dst
    Performs initial step of meanshift segmentation of an image.

    cv.pyrUp(src[, dst[, dstsize[, borderType]]]) ->dst
    Upsamples an image and then blurs it.

    cv.Scharr(src, ddepth, dx, dy[, dst[, scale[, delta[, borderType]]]]) ->dst
    Calculates the first x- or y- image derivative using Scharr operator.

    cv.sepFilter2D(src, ddepth, kernelX, kernelY[, dst[, anchor[, delta[, borderType]]]]) ->dst
    Applies a separable linear filter to an image.

    cv.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]) ->dst
    Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator.

    cv.spatialGradient(src[, dx[, dy[, ksize[, borderType]]]]) ->dx, dy
    Calculates the first order image derivative in both x and y using a Sobel operator.

    cv.sqrBoxFilter(src, ddepth, ksize[, dst[, anchor[, normalize[, borderType]]]]) ->dst
    Calculates the normalized sum of squares of the pixel values overlapping the filter.

    cv.stackBlur(src, ksize[, dst]) ->dst
    Blurs an image using the stackBlur.


    The environment is set up with the necessary libraries and the checkpoints are downloaded (if there are any).
    """
    return task_message

code_verifier_system_message = """
You are an experienced coder who acts as a code verifier. Your task is to clean up code written by other coders. When presented with code output:
Extract only the actual executable code, removing any thought processes, explanations, or duplicate code blocks.
Keep the comments or docstrings that are relevant to the code.
Present the code as a single, clean Python code block that is ready to execute.
Remember we are using a stateful IPython kernel where variables from previous code blocks are accessible and 
bash commands are supported too.

Your response should be either:
A single Python code block containing the verified code (bash command is also considered code block), or
"NO_CODE" if there is no executable code.
"""


code_writer_system_message = """
You are an experienced coder tasked with figuring out the best way to preprocess, i.e. transform, an image such
that a pretrained detection model can better identify spots in an image.
You have access to the source code repo and can interactively read files to understand the implementation.
The environment is set up with the necessary libraries and the checkpoints are downladed (if there are any).

You write code using Python in a stateful IPython kernel, where:
- Write code in Python markdown code blocks:
```python
x = 3  # Code example
```
- Use previously defined variables in new code blocks:
```python
print(x)  # Using previous variable
```
- Use bash commands with prefix `!` to explore and interact with files:
```python
!ls  # List files
!cat file.py  # View file contents
```
 (if you encounter a .ipynb file, you might want to convert it into .py file before viewing)
- Execute any other bash commands to navigate and understand the codebase 
    - README.md is usually the best place to start
    - If you have doubts, check documentation, examples or source code files for more information
- Code outputs will be returned to you
- For generated images, you'll receive the file path
- Write code incrementally to build your solution
- Write "TERMINATE" when the task is complete

Your result will be verified by another coder so you can write down your thought process and exploration steps if needed.

The execution result will be returned to you and when it fails, you should analyze the error message and either make targeted 
changes if the fix is clear, or return to the repository to gather more information if the issue is not immediately obvious or 
if you stumble upon a dead end.

The spot detection results will be evaluated by with a loss function to indicate how well spots were detected. Given the feedback,
you may need to revise your code. Reflect on the feedback, explore the repo to see if the issue can be fixed, and update your code accordingly.
"""