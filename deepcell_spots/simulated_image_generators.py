"""
Simulated image generators -
Generate images of simple objects for testing of object detection pipelines
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import multivariate_normal
from skimage import draw
import skimage


def is_in_image(x,y,a,L):
    # return True if the square with left bottom corner at (x,y) and side length a
    # is contained in [0,L-1]X[0,L-1]
    if (x+a<=(L-1)) and (y+a<=(L-1)) and (x>=0) and (y>=0):
        return True
    else:
        return False

def is_in_square(x,y,a,x1,y1):
    # returns True if (x1,y1) is inside the square with left bottom corner at (x,y) and side length a
    if (x<=x1<=x+a) and (y<=y1<=y+a):
        return True
    else:
        return False 
    
def is_overlapping(x_list,y_list,side_length_list,x,y,a):
    # check if a square with left corner at x,y,a
    # overlaps with other squares with corner coordinates and side length in the list
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    side_length_list = np.array(side_length_list)
    not_overlapping = ((x+a)<x_list) | ((x_list+side_length_list)<x) | ((y+a)<y_list) | ((y_list+side_length_list)<y)
    return not all(not_overlapping)

def add_gaussian_noise(image,m,s):
    ''' Adds gaussian random noise with mean m and standard deviation s to the input image
    Args:
    image: 2D np.array
    m: mean of gaussian random noise to be added to each pixel of image
    s: standard deviation of gaussian random noise to be added to each pixel of image
    '''
    row,col= image.shape
    gauss = np.random.normal(m,s,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    return noisy

def generate_nonoverlapping_square_img(L,N_min,N_max):
    '''
    Generates an L*L image with pixel values in [0,1] which has a background of 0s and non-overlapping squares
    with various intensities.
    
    Args:
        L: length in pixels of image side
        N_min: integer minimal number of squares to be drawn on the image
        N_max: integer maximal number of squares to be drawn on the image
            The number of generated squares is drawn uniformly from the set of integers in [Nmin Nmax]
            
    Returns:
        img: The generated image of squares with a background of 0s
    '''
       
    # length of square side: vary uniformly in a range
    a_min = 1
    a_max = 100
    
    # intensity - assume uniform distribution
    A_min = 0.2
    A_max = 1.0
    
    # create the image
    img = np.zeros((L,L))
    # coordinates for grid of pixels
    X = np.arange(0,L,1) + 0.5
    Y = np.arange(0,L,1) + 0.5
    X, Y = np.meshgrid(X, Y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    
    # draw the number of objects that will be created in the image - an integer N_min <= N <= N_max
    N = random.randint(N_min, N_max)
    
    
    # allocate variables to store dot positions
    x_list = []
    y_list = []
    side_length_list = []
    # loop on all squares, draw each square until it does not overlap with previous ones, then draw its intensity and add it to img
    for ind in range(N):
        # draw the position (x,y) of the lower left corner of the square uniformly from [0,L-1]*[0,L-1]
        x = random.randint(0, L-1)
        y = random.randint(0, L-1)
    
        # draw the side length of the square
        a = random.randint(a_min, a_max)
        
        # check if square overlaps with others
        while ((not is_in_image(x,y,a,L)) or is_overlapping(x_list,y_list,side_length_list,x,y,a)):
            # draw the position (x,y) of the lower left corner of the square uniformly from [0,L-1]*[0,L-1]
            x = random.randint(0, L-1)
            y = random.randint(0, L-1)
        
            # draw the side length of the square
            a = random.randint(a_min, a_max)
        
        # draw the intensity from a uniform distribution
        A = random.uniform(A_min, A_max)
            
        # add new square properties to lists
        x_list.append(x)
        y_list.append(y)
        side_length_list.append(a)
        
        # add the square to the image
        in_square = (X>=x) & (X<=x+a) & (Y>=y) & (Y<=y+a)
        img[in_square] += A
        
    #img = add_gaussian_noise(img, 0, 0.002)
    
    return img


def generate_nonoverlapping_square_img_with_segmask(L,N_min,N_max):
    '''
    Generates an L*L image with pixel values in [0,1] which has a background of 0s and non-overlapping squares
    with various intensities.
    
    Args:
        L: length in pixels of image side
        N_min: integer minimal number of squares to be drawn on the image
        N_max: integer maximal number of squares to be drawn on the image
            The number of generated squares is drawn uniformly from the set of integers in [Nmin Nmax]
            
    Returns:
        img: The generated image of squares with a background of 0s
        segmask: The segmentation mask for the image: background=0, each object's pixels are marked with value i for i=1,...,N for N=number of objects in the image
    '''

    
    # simulated data:
    # grascale image L*L pixels of N squares at random positions with varying intensities
    
    #L = 256
    
    # parameters
    # number of squares: vary uniformly in a range
    #N_min = 1
    #N_max = 10
    
    # length of square side: vary uniformly in a range
    a_min = 1
    a_max = 100
    
    # intensity - assume uniform distribution
    A_min = 0.2
    A_max = 1.0
    
    # create the image
    img = np.zeros((L,L))
    segmask = np.zeros((L,L)) # create the matching segmentation mask
    # coordinates for grid of pixels
    X = np.arange(0,L,1) + 0.5
    Y = np.arange(0,L,1) + 0.5
    X, Y = np.meshgrid(X, Y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    
    # draw the number of objects that will be created in the image - an integer N_min <= N <= N_max
    N = random.randint(N_min, N_max)
    
    
    # allocate variables to store square positions
    x_list = []
    y_list = []
    side_length_list = []
    # loop on all squares, draw each square until it does not overlap with previous ones, then draw its intensity and add it to img
    for ind in range(N):
        # draw the position (x,y) of the lower left corner of the square uniformly from [0,L-1]*[0,L-1]
        x = random.randint(0, L-1)
        y = random.randint(0, L-1)
    
        # draw the side length of the square
        a = random.randint(a_min, a_max)
        
        # check if square overlaps with others
        while ((not is_in_image(x,y,a,L)) or is_overlapping(x_list,y_list,side_length_list,x,y,a)):
            # draw the position (x,y) of the lower left corner of the square uniformly from [0,L-1]*[0,L-1]
            x = random.randint(0, L-1)
            y = random.randint(0, L-1)
        
            # draw the side length of the square
            a = random.randint(a_min, a_max)
        
        # draw the intensity from a uniform distribution
        A = random.uniform(A_min, A_max)
            
        # add new square properties to lists
        x_list.append(x)
        y_list.append(y)
        side_length_list.append(a)
        
        # add the square to the image and to the segmentation mask
        in_square = (X>=x) & (X<=x+a) & (Y>=y) & (Y<=y+a)
        img[in_square] += A
        segmask[in_square] = ind+1
        
    #img = add_gaussian_noise(img, 0, 0.002)
    
    return (img, segmask)


def uniform_dist_size_nonoverlapping_square_img_generator(L,N_max,a_min,a_max,noise=False):
    '''
    Generates L*L images with pixel values in [0,1] which has a background of 0s and non-overlapping squares
    with various intensities.
    The side length of generated squares is drawn uniformly from the set of integers in [a_min a_max].
    A position for the square is drawn randomly, and redrawn up to a max_trial amount of times until a position
    in which it doesn't overlap with existing squares in the image is successfully drawn
    If after max_trial redraws the square still overlaps with existing squares, the image is returned and
    the square is saved for the next image
    This way, the square sizes over all the images are distributed uniformly, while the distribution of number
    of squares is probably not uniform
    
    Args:
        L: length in pixels of image side
        a_min: the minimal square side length to be generated
        a_max: the maximal square side length to be generated
        N_max: integer maximal number of squares to be drawn on the image
        noise: if True, white noise will be added to the image
            
    Returns:
        img: The generated image of squares with a background of 0s
        segmask: The segmentation mask for the image: background=0, each object's pixels are marked with value i for i=1,...,N for N=number of objects in the image
    '''
    # parameters for simulated data:
    # grascale image L*L pixels of N squares at random positions with varying intensities
    
    #L = 256
    
    # parameters
    # number of squares: vary uniformly in a range
    #N_min = 1
    #N_max = 10
    
    # length of square side: vary uniformly in a range
    #a_min = 1
    #a_max = 100
    
    resample_max = 10 # maximal number of times to redraw object position before saving it for plotting in the next image
    
    # intensity - assume uniform distribution
    A_min = 0.2
    A_max = 1.0 
    
    noise_std = 0.01 # white noise standard deviation
    
    # coordinates for grid of pixels
    X = np.arange(0,L,1)
    Y = np.arange(0,L,1)
    X, Y = np.meshgrid(X, Y)
    
    while True: 
        # create a new image
        img = np.zeros((L,L))
        segmask = np.zeros((L,L)) # create the matching segmentation mask

        N = 0 # number of objects plotted to the image    
        resample_num = 0 # number of times object position was redrawn
        # allocate variables to store square positions in a single image
        x_list = []
        y_list = []
        side_length_list = []
        # loop on all squares, draw each square until it does not overlap with previous ones, then draw its intensity and add it to img
        while N<=N_max:
            # draw the side length of the square
            a = random.randint(a_min, a_max)
            
            # draw the position (x,y) of the lower left corner of the square uniformly from [0,L-a-1]*[0,L-a-1] (uniformly inside image boundaries)
            x = random.randint(0, L-a-1)
            y = random.randint(0, L-a-1)
            
            # check if square overlaps with others, if so resample the lower left corner position form a uniform distribution in the image
            while resample_num<=resample_max and is_overlapping(x_list,y_list,side_length_list,x,y,a):
                # draw the position (x,y) of the lower left corner of the square uniformly from [0,L-1]*[0,L-1]
                x = random.randint(0, L-a-1)
                y = random.randint(0, L-a-1)
                
                resample_num += 1 # increase number of samples counter by 1               
                
            if is_overlapping(x_list,y_list,side_length_list,x,y,a): # done resampling, still haven't found an empty space for this square
                if noise==True: # add white noise to the image
                    img = add_gaussian_noise(img, 0, noise_std)
                    img[img<0] = 0 # when white noise was added to 0 background, negative pixel values are obtained - set them to zero
                yield (img, segmask)
                
                # create a new image
                img = np.zeros((L,L))
                segmask = np.zeros((L,L)) # create the matching segmentation mask
                
                N = 0 # number of objects plotted to the image    
                resample_num = 0 # number of times object position was redrawn
                # allocate variables to store square positions in a single image
                x_list = []
                y_list = []
                side_length_list = []
    
            # if found a place for this square, save it to this image
            # draw the intensity from a uniform distribution
            A = random.uniform(A_min, A_max)
                
            # add new square properties to lists
            x_list.append(x)
            y_list.append(y)
            side_length_list.append(a)
            
            # add the square to the image and to the segmentation mask
            in_square = (X>=x) & (X<=x+a) & (Y>=y) & (Y<=y+a)
            img[in_square] += A
            segmask[in_square] = N+1
            N += 1 # increase number of objects in image counter by 1
         
        
        # reached the maximal number of squares in a single image - save it and start a new one
        if noise==True: # add white noise to the image
            img = add_gaussian_noise(img, 0, noise_std)
            img[img<0] = 0 # when white noise was added to 0 background, negative pixel values are obtained - set them to zero
        yield (img, segmask)
        

def is_circle_overlapping(x_list,y_list,r_list,x,y,r):
    # check if a circle with center at x,y and radius r
    # overlaps with other circles with center coordinates and radius on the list
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    r_list = np.array(r_list)
    dist_sq = (x_list-x)**2 + (y_list-y)**2
    overlapping = dist_sq < (r_list+r)**2
    return any(overlapping)

def uniform_dist_size_nonoverlapping_circle_img_generator(L,N_max,r_min,r_max,noise=False):
    '''
    Generates L*L images with pixel values in [0,1] which has a background of 0s and non-overlapping squares
    with various intensities.
    The radius of the circles is drawn uniformly from the set of integers in [r_min r_max].
    A position for the circle is drawn randomly, and redrawn up to a max_trial amount of times until a position
    in which it doesn't overlap with existing squares in the image is successfully drawn
    If after max_trial redraws the circle still overlaps with existing squares, the image is returned and
    the square is saved for the next image
    This way, the square sizes over all the images are distributed uniformly, while the distribution of number
    of circles is probably not uniform
    
    Args:
        L: length in pixels of image side
        r_min: the minimal square radius to be generated
        r_max: the maximal square radius to be generated
        N_max: integer maximal number of squares to be drawn on the image
        noise: if True, white noise will be added to the image
            
    Returns:
        img: The generated image of squares with a background of 0s
        segmask: The segmentation mask for the image: background=0, each object's pixels are marked with value i for i=1,...,N for N=number of objects in the image
    '''
    # parameters for simulated data: 
    resample_max = 10 # maximal number of times to redraw object position before saving it for plotting in the next image
    
    # intensity - assume uniform distribution
    A_min = 0.2
    A_max = 1.0
    
    noise_std = 0.01 # white noise standard deviation
    
    # coordinates for grid of pixels
    X = np.arange(0,L,1)
    Y = np.arange(0,L,1)
    X, Y = np.meshgrid(X, Y)
    
    while True: 
        # create a new image
        img = np.zeros((L,L))
        segmask = np.zeros((L,L)) # create the matching segmentation mask

        N = 0 # number of objects plotted to the image    
        resample_num = 0 # number of times object position was redrawn
        # allocate variables to store square positions in a single image
        x_list = []
        y_list = []
        r_list = []
        # loop on all squares, draw each square until it does not overlap with previous ones, then draw its intensity and add it to img
        while N<=N_max:
            # draw the radius of the circle
            r = random.randint(r_min, r_max)
            # draw the position (x,y) of the circle center uniformly s.t. the circle is entirely inside the image
            x = random.randint(r, L-r-1)
            y = random.randint(r, L-r-1)
            
            # check if square overlaps with others, if so resample the lower left corner position form a uniform distribution in the image
            while resample_num<=resample_max and is_circle_overlapping(x_list,y_list,r_list,x,y,r):
                # draw the position (x,y) of the circle center uniformly s.t. the circle is entirely inside the image
                x = random.randint(r, L-r-1)
                y = random.randint(r, L-r-1)
                
                resample_num += 1 # increase number of samples counter by 1               
                
            if is_circle_overlapping(x_list,y_list,r_list,x,y,r): # done resampling, still haven't found an empty space for this square
                if noise==True: # add white noise to the image
                    img = add_gaussian_noise(img, 0, noise_std)
                    img[img<0] = 0 # when white noise was added to 0 background, negative pixel values are obtained - set them to zero
                yield (img, segmask)
                
                # create a new image
                img = np.zeros((L,L))
                segmask = np.zeros((L,L)) # create the matching segmentation mask
                
                N = 0 # number of objects plotted to the image    
                resample_num = 0 # number of times object position was redrawn
                # allocate variables to store square positions in a single image
                x_list = []
                y_list = []
                r_list = []
    
            # if found a place for this square, save it to this image
            # draw the intensity from a uniform distribution
            A = random.uniform(A_min, A_max)
                
            # add new square properties to lists
            x_list.append(x)
            y_list.append(y)
            r_list.append(r)
            
            # add the circle to the image and to the segmentation mask
            #in_square = (X>=x) & (X<=x+a) & (Y>=y) & (Y<=y+a)
            #img[in_square] += A
            #segmask[in_square] = N+1
            rr, cc = skimage.draw.circle(x, y, r, shape=img.shape)
            img[rr,cc] += A
            segmask[rr,cc] = N+1
            N += 1 # increase number of objects in image counter by 1
         
        
        # reached the maximal number of squares in a single image - save it and start a new one
        if noise==True: # add white noise to the image
            img = add_gaussian_noise(img, 0, noise_std)
            img[img<0] = 0 # when white noise was added to 0 background, negative pixel values are obtained - set them to zero
        yield (img, segmask)
           
        
        

def gaussian_spot_image_generator(L, N_min, N_max, sigma_mean, sigma_std, A_mean=1, A_std=0,noise_mean=0,noise_std=0,segmask=False,yield_pos=False):
    ''' Generates random images of gaussian spots with random uniformly distributed center positions in the image area, 
    i.e. in [0,L-1]*[0,L-1]. The number of spots in an image is uniformly distributed in [N_min, N_max]. Each spot is a gaussian
    with standard deviation normally distributed with mean sigma0, std sigma1, and cutoff value of 0.5 (it is redrawn if a smaller value is drawn).
    The intensity of each spot is normally distributed.
    
    Args:
        L : generated image side length - the generated images have shape (L,L)
        N_min, N_max: the number of spots plotted in each image is uniformly distributed in [N_min, N_max]
        sigma0, sigma1: the mean and standard deviation of the normally distributed spot width sigma (i.e. each spot is a gaussian with standard deviation sigma)
        A_mean, A_std: the intensity of each spot is normally distributed in with mean A_mean, and standard deviation A_std
        yield_pos: if True, will yield lists of x and y positions and bounding boxes in addition to image and label image
        noise_mean, noise_std: mean and std of white noise to be added to every pixel of the image
        
    Yields:
        img : (L,L) numpy array simulated image
        label : (L,L) numpy array of - 0 background, 1 for pixel of (rounded) spot center if segmask==False
                                       segmentation mask if segmask==True (pixel values are 0 in background, 1,...,N for pixels belonging to the N spots in the image)
    '''
       
    while True: # keep yielding images forever
        img = np.zeros((L,L))     # create the image
    
        # coordinates for grid of pixels
        X = np.arange(0,L,1)
        Y = np.arange(0,L,1)
        X, Y = np.meshgrid(X, Y)
        #pos = np.empty(X.shape + (2,))
        #pos[:, :, 0] = X
        #pos[:, :, 1] = Y
        
        # draw the number of dots that will be created in the image - an integer N_min <= N <= N_max
        N = random.randint(N_min, N_max)
        # allocate variables to store dot positions
        x_list = []
        y_list = []
        sigma_list = []
        bboxes = []
        # loop on all dots, generate the intensity for each dot and sum into img
        for ind in range(N):
            # draw the position (x,y) uniformly from [-0.5,L-0.5]*[-0.5,L-0.5]
            x = random.uniform(-0.5, L-0.5)
            y = random.uniform(-0.5, L-0.5)
            x_list.append(x)
            y_list.append(y)
            
            # draw the width of the gaussian
            sigma = random.gauss(sigma_mean, sigma_std)
            #while sigma < 0.5: # if sigma is too small, resample because it causes divergence
            #    sigma = random.gauss(sigma_mean, sigma_std)
            
            sigma_list.append(sigma)
            # draw the intensity from a normal distribution
            A = random.gauss(A_mean, A_std)
            
            # add a the bounding box inscribing the circle of radius sigma and center in (x,y) to the list
            x1 = x - sigma
            x2 = x + sigma
            y1 = y - sigma
            y2 = y + sigma
            bboxes.append([x1, y1, x2, y2])
            
            # plot a gaussian
            # Mean vector and covariance matrix
            #mu = np.array([x, y])
            #sigma = np.array([[sigma , 0], [0,  sigma]])
            #Z = A*multivariate_normal.pdf(pos,mean=mu, cov=sigma)
            Z = A*np.exp(-((X-x)**2+(Y-y)**2)/(2*sigma**2))
            img += Z
            
            # add white noise to the image
            img = add_gaussian_noise(img, noise_mean, noise_std)
            img[img<0] = 0 # when white noise was added to 0 background, negative pixel values are obtained - set them to zero
            
        # create label
        if segmask==True:
            # create segmentation mask
            label = np.zeros((L,L))     # create the image
            for spot_ind in range(len(x_list)):
                rr, cc = skimage.draw.circle(x_list[spot_ind], y_list[spot_ind], sigma_list[spot_ind], shape=label.shape)
                label[cc,rr] = spot_ind+1
        else:
            # create image with background 0, spot centers labeled with 1
            # use floor function since pixel with indices (i,j) covers the area [i,i+1] x [j,j+1] and thus contains all coordinates of the form (i.x,j.x)
            x_ind = np.floor(x_list)
            y_ind = np.floor(y_list)
            label = np.zeros((L,L))     # create the image
            label[y_ind.astype(int),x_ind.astype(int)] = 1 # pixels that are at the (rounded) center of a dot, are marked with 1
            
        if yield_pos==False:
            yield (img, label)
        else:
            bboxes = np.array(bboxes)
            bboxes = np.reshape(bboxes, (bboxes.shape[0], 4)) # reshape bboxes in case it is empty.
            yield (img, label, x_list, y_list, bboxes)
