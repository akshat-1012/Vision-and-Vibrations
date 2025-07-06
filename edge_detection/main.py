import numpy as np
from matplotlib import pyplot as plt
import cv2
from scipy.signal import correlate2d
from grayscaling import to_grayscale
from nonmaximumsuppression import non_max_suppression
from dth_and_hysteresis import threshold_hysteresis

def scale(x):
    '''scale between 0 and 255'''
    return (x - x.min()) / (x.max() - x.min()) * 255\

def gradiantCalculation(img_blur):
    Kx = np.array(
        [[-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]], np.float32
    )

    Ky = np.array(
        [[-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]], np.float32
    )

    Gradient_Y = correlate2d(img_blur, Ky)
    Gradient_X = correlate2d(img_blur, Kx)
    G = scale(np.hypot(Gradient_X, Gradient_Y))
    return G

def thetaCalculation(img_blur):
    Kx = np.array(
        [[-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]], np.float32
    )

    Ky = np.array(
        [[-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]], np.float32
    )

    Gradient_Y = correlate2d(img_blur, Ky)
    Gradient_X = correlate2d(img_blur, Kx)
    theta = np.arctan2(Gradient_Y, Gradient_X)
    return theta

if __name__ == "__main__":
    #read Image
    img = cv2.imread(r"/Users/akshatsrivastava/Desktop/VnV/edgeDet.png")

    #gray scale
    gray_image = to_grayscale(img)      # We could have also done gray_image= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #gaussian blurring to reduce noise present in the image for better edge detection
    blurred_image = cv2.GaussianBlur(src=gray_image, ksize=(5, 5), sigmaX=3)

    #Gradient Calculation using Sobel Filter,
    G = gradiantCalculation(blurred_image)
    theta = thetaCalculation(blurred_image)

    #Non Maximum suppression
    Z = non_max_suppression(G,theta)

    res = threshold_hysteresis(Z)


    res_uint8 = res.astype(np.uint8) #cvt.COLOR expects uint8 datatype

    #Display the final image
    plt.imshow(cv2.cvtColor(res_uint8, cv2.COLOR_BGR2RGB))
    plt.show()
    

