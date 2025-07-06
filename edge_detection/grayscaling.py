import numpy as np;


def to_grayscale(img: np.ndarray):
    b,g,r = img[... , 0] , img[..., 1], img[..., 2]
    grayscale_image = 0.1140*b + 0.5870*g + 0.2989*r
    return grayscale_image