import numpy as np
from scipy.ndimage import convolve

def calcIxIyIt(img1, img2):
    #derivative masks
    x_kernel = np.array([[-1, 1], [-1, 1]]) 
    y_kernel = np.array([[1, 1], [-1, -1]])
    # t_kernel = np.ones((2, 2)) 

    Ix = (convolve(img1,x_kernel) + convolve(img2,x_kernel))*0.5
    Iy = (convolve(img1,y_kernel) + convolve(img2,y_kernel))*0.5
    # It = convolve(img1, -t_kernel) + convolve(img2, t_kernel)
    It = img2 - img1

    return [Ix,Iy,It]