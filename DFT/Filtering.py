# For this part of the assignment, You can use inbuilt functions to compute the fourier transform
# You are welcome to use fft that are available in numpy and opencv
import cv2
import scipy.misc
import numpy as np
import math
from matplotlib import pyplot as plt
from PIL import Image

class Filtering:
    image = None
    filter = None
    cutoff = None
    order = None

    def __init__(self, image, filter_name, cutoff, order = 0):
        """initializes the variables frequency filtering on an input image
        takes as input:
        image: the input image
        filter_name: the name of the mask to use
        cutoff: the cutoff frequency of the filter
        order: the order of the filter (only for butterworth
        returns"""
        self.image = image
        if filter_name == 'ideal_l':
            self.filter = self.get_ideal_low_pass_filter
        elif filter_name == 'ideal_h':
            self.filter = self.get_ideal_high_pass_filter
        elif filter_name == 'butterworth_l':
            self.filter = self.get_butterworth_low_pass_filter
        elif filter_name == 'butterworth_h':
            self.filter = self.get_butterworth_high_pass_filter
        elif filter_name == 'gaussian_l':
            self.filter = self.get_gaussian_low_pass_filter
        elif filter_name == 'gaussian_h':
            self.filter = self.get_gaussian_high_pass_filter

        self.cutoff = cutoff
        self.order = order


    def get_ideal_low_pass_filter(self, shape, cutoff,order):
        """Computes a Ideal low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal low pass mask"""
        w = shape[0]
        h = shape[1]

        A = np.ones((w,h))
        c1 = w/2
        c2 = h/2

        for i in range(1,w):
            for j in range(1,h):
                r1 = (i-c1)** 2+(j-c2)**2
                r= math.sqrt(r1)

                if(r > cutoff):
                    A[i,j] = 0.0

        return A


    def get_ideal_high_pass_filter(self, shape, cutoff, order):
        """Computes a Ideal high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask
        w = shape[0]
        h = shape[1]

        A = np.ones((w, h))
        c1 = w / 2
        c2 = h / 2

        for i in range(1, w):
            for j in range(1, h):
                r1 = (i - c1) ** 2 + (j - c2) ** 2
                r = math.sqrt(r1)

                if (0< r < cutoff):
                    A[i, j] = 0.0

        return A


    def get_butterworth_low_pass_filter(self, shape, cutoff, order):
        """Computes a butterworth low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth low pass mask"""
        w = shape[0]
        h = shape[1]

        A = np.ones((w, h))
        c1 = w / 2
        c2 = h / 2

        for i in range(1, w):
            for j in range(1, h):
                r1 = (i - c1) ** 2 + (j - c2) ** 2
                r = math.sqrt(r1)

                if (r > cutoff):
                    A[i, j] = 1 / (1 + (r / cutoff) ** order)

        return A


    def get_butterworth_high_pass_filter(self, shape, cutoff, order):
        """Computes a butterworth high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask
        w = shape[0]
        h = shape[1]

        A = np.ones((w, h))
        c1 = w / 2
        c2 = h / 2

        for i in range(1, w):
            for j in range(1, h):
                r1 = (i - c1) ** 2 + (j - c2) ** 2
                r = math.sqrt(r1)

                if (0 < r < cutoff):
                    A[i, j] = 1 / (1 + (r / cutoff) ** order)

        return A

    def get_gaussian_low_pass_filter(self, shape, cutoff, order):
        """Computes a gaussian low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian low pass mask"""
        w = shape[0]
        h = shape[1]

        A = np.ones((w, h))
        c1 = w / 2
        c2 = h / 2

        t1 = 2*cutoff

        for i in range(1, w):
            for j in range(1, h):
                r1 = (i - c1) ** 2 + (j - c2) ** 2
                r = math.sqrt(r1)

                if (r > cutoff):
                    A[i, j] = math.exp(-r**2/t1**2)

        return A

    def get_gaussian_high_pass_filter(self, shape, cutoff, order):
        """Computes a gaussian high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask
        w = shape[0]
        h = shape[1]

        A = np.ones((w, h))
        c1 = w / 2
        c2 = h / 2

        t1 = 2 * cutoff

        for i in range(1, w):
            for j in range(1, h):
                r1 = (i - c1) ** 2 + (j - c2) ** 2
                r = math.sqrt(r1)

                if (0 < r < cutoff):
                    A[i, j] = 1- math.exp(-r ** 2 / t1 ** 2)

        return A

    def post_process_image(self, image):
        """Post process the image to create a full contrast stretch of the image
        takes as input:
        image: the image obtained from the inverse fourier transform
        return an image with full contrast stretch
        -----------------------------------------------------
        1. Full contrast stretch (fsimage)
        2. take negative (255 - fsimage)
        """

        x = image.shape[0]
        y = image.shape[1]

        min = image.min()
        max = image.max()

        for i in range(1,x):
            for j in range(1,y):
                image[i,j] = ((image[i,j] - min)/(max-min))*255

        return image


    def filtering(self):
        """Performs frequency filtering on an input image
        returns a filtered image, magnitude of DFT, magnitude of filtered DFT        
        ----------------------------------------------------------
        You are allowed to used inbuilt functions to compute fft
        There are packages available in numpy as well as in opencv
        Steps:
        1. Compute the fft of the image
        2. shift the fft to center the low frequencies
        3. get the mask (write your code in functions provided above) the functions can be called by self.filter(shape, cutoff, order)
        4. filter the image frequency based on the mask (Convolution theorem)
        5. compute the inverse shift
        6. compute the inverse fourier transform
        7. compute the magnitude
        8. You will need to do a full contrast stretch on the magnitude and depending on the algorithm you may also need to
        take negative of the image to be able to view it (use post_process_image to write this code)
        Note: You do not have to do zero padding as discussed in class, the inbuilt functions takes care of that
        filtered image, magnitude of DFT, magnitude of filtered DFT: Make sure all images being returned have grey scale full contrast stretch and dtype=uint8 
        """
        image = self.image
        image =np.asarray(image)

        fft_image = np.fft.fft2(image)
        fft_shift = np.fft.fftshift(fft_image)
        magnitude_DFT = np.log(np.abs(fft_shift)).astype(np.uint8)


        w = fft_shift.shape[0]
        h = fft_shift.shape[1]

        mask = self.filter((w,h), self.cutoff,self.order)

        #covolution function
        con = fft_shift * mask
        filtered_dft = np.log(np.abs(con)).astype(np.uint8)


        inverse = np.fft.ifftshift(con)
        inversefft = np.fft.ifft2(inverse)

        filtered_image = np.log(np.abs(inversefft))
        filtered_image = self.post_process_image(filtered_image)


        return [filtered_image, self.post_process_image(magnitude_DFT), self.post_process_image(filtered_dft)]
