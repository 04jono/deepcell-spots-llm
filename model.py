from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-70B-Instruct")
pipe = pipeline("text-generation", "meta-llama/Llama-3.1-70B-Instruct", tokenizer=tokenizer)

def generate_text(prompt):
    messages = [
        {"role": "system", "content": '''
                    You will write a function that transforms an image such that it will
                    be easier to detect spots in it. You will be prompted with example functions.
                    Do not write the same type of math function as the example functions.
                    Only return the function itself. The function must have
                    two arguments: image (numpy.array) and clip (boolean).
         '''},
         {
             "role": "system", "content": '''
                    Here are OpenCV image processing functions you can use:
                    
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


        '''
         },
        {"role": "user", "content": prompt},
    ]    
    outputs = pipe(messages, return_full_text=False, max_new_tokens=512) 
    
    return outputs[0]["generated_text"]

def read_file_to_string(file_path):
    with open(file_path, 'r') as file:
        file_contents = file.read()
    return file_contents


function_bank = read_file_to_string('function_bank.py')

model_output = str(generate_text(function_bank))

generated_function = model_output.split('```python')[1].split('```')[0]

print(generated_function)

#Write the generated function to a json file
import json

with open('function_bank.json', 'r') as file:
    json_array = json.load(file)

with open('function_bank.json', 'w') as file:

    class_score = 0
    regress_score = 0
    json_data = {
        "code": generated_function,
        "class_score": class_score,
        "regress_score": regress_score
    }

    json_array.append(json_data)

    json.dump(json_array, file)
