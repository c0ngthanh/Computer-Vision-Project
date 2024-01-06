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
def rgb2gray(rgb: np.array) -> np.array:
    """
    Returns gray image
    :param img: RGB image
    """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
def color2gray(img: np.ndarray) -> np.ndarray:
    """
    Returns gray image with preserved gradients.
    :param img: RGB image
    """
    img_h, img_w = img.shape[:2]
    img_s = img.copy()
    img_t = rgb2gray(img.copy())
    
    im2var = np.arange(img_h * img_w).reshape(img_h, img_w) 
    
    A = sp.sparse.lil_matrix((img_h*img_w*4, img_h*img_w))
    b = np.zeros(img_h*img_w*4)
    
    e = 0
    for y in range(img_h):
        for x in range(img_w):
            dr = abs(img_s[y][x][0])
            dg = abs(img_s[y][x][1])
            db = abs(img_s[y][x][2])
            
            A[e, im2var[y][x]] = 1
            b[e] = sum([img_s[y][x][n] for n in range(3)]) / 3
            e += 1
                    
    A = sp.sparse.csr_matrix(A)
    v = sp.sparse.linalg.lsqr(A, b)[0]
    
    return v.reshape(img_h, img_w)

def run():
    # Load Image and set up parameters
    color_blind_img = helper.get_image('samples/img5.jpg', scale=False)
    color2gray_img = color2gray(color_blind_img)

    #Run function, don't do anything
    helper.show_images(
    [color_blind_img, cv2.cvtColor(color_blind_img, cv2.COLOR_BGR2GRAY), color2gray_img], 
    ["RGB image", "Gray image (openCV)", "Gray image (color2gray)"], 
    figsize=(12, 8)
    )
run()