import warnings
warnings.filterwarnings('ignore')
import cv2
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import os
from random import random
import time
import scipy as sp
import scipy.sparse.linalg
from typing import List, Tuple
import helper
def poisson_sharpening(
    img: np.ndarray, 
    alpha: int
) -> np.ndarray:
    """
    Returns a shapren image with strenght of alpha.
    :param img: the image
    :param alpha: edge threshold and gradient scaler
    """
    img_h, img_w = img.shape[:2]
    img_s = img.copy()
    
    im2var = np.arange(img_h * img_w).reshape(img_h, img_w) 
    
    A = sp.sparse.lil_matrix((img_h*img_w*4*2, img_h*img_w))
    b = np.zeros(img_h*img_w*4*2)
    
    e = 0
    for y in range(img_h):
        for x in range(img_w):
            A[e, im2var[y][x]] = 1
            b[e] = img_s[y][x]
            e += 1
            
            for n_y, n_x in helper.neighbours(y, x, img_h-1, img_w-1):
                A[e, im2var[y][x]] = 1
                A[e, im2var[n_y][n_x]] = -1
                
                b[e] = alpha * (img_s[y][x] - img_s[n_y][n_x])
                e += 1
                
    A = sp.sparse.csr_matrix(A)
    v = sp.sparse.linalg.lsqr(A, b)[0]

    return np.clip(v.reshape(img_h, img_w), 0, 1)
def run():
    # Load Image and set up parameters
    img = helper.get_image('samples/img6.jpg')
    alpha = 9.0

    #Run function, don't do anything
    sharpen_img = np.zeros(img.shape)
    for b in np.arange(3):
        sharpen_img[:,:,b] = poisson_sharpening(img[:,:,b], alpha)
    helper.show_images(
    [img, sharpen_img], 
    ["Original image", "Sharpen image"], 
    figsize=(8, 8)
    )
run()