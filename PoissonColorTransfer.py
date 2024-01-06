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
def color_transfer(
    img_s: np.ndarray, 
    img_t: np.ndarray,
    alpha: int
) -> np.ndarray:
    """
    Returns img_t with color of img_s and strenght of alpha.
    :param img_s: the source image with color style
    :param img_t: the target image to transfer color to
    :param alpha: edge threshold and gradient scaler
    """
    img_t_h, img_t_w = img_t.shape[:2]
        
    im2var = np.arange(img_t_h * img_t_w * 3).reshape((img_t_h, img_t_w, 3)) 
    
    A = sp.sparse.lil_matrix((img_t_h*img_t_w*3*5, img_t_h*img_t_w*3))
    b = np.zeros(img_t_h*img_t_w*3*5)
    
    e = 0
    for n in range(3):
        img_s_channel_avg = np.mean(img_s[:,:,n])
        img_s_channel_std = np.std(img_s[:,:,n])
        img_t_channel_avg = np.mean(img_t[:,:,n])
        img_t_channel_std = np.std(img_t[:,:,n])
        
        for y in range(img_t_h):
            for x in range(img_t_w):
                A[e, im2var[y][x][n]] = 1

                b[e] = img_t[y][x][n] - (img_t_channel_avg * (img_t_channel_std / img_s_channel_std) - img_s_channel_avg) * alpha
                e += 1
                
                for n_y, n_x in helper.neighbours(y, x, img_t_h-1, img_t_w-1):
                    A[e, im2var[y][x][n]] = 1
                    A[e, im2var[n_y][n_x][n]] = -1
                    b[e] = img_t[y][x][n] - img_t[n_y][n_x][n]
                    e += 1
    
    A = sp.sparse.csr_matrix(A)
    v = sp.sparse.linalg.lsqr(A, b)[0]
      
    return np.clip(v.reshape((img_t_h, img_t_w, 3)), 0, 1)
def run():
    # Load Image and set up parameters
    style_img = helper.get_image('samples/img7.jpg')
    target_img = helper.get_image('samples/img8.jpg')
    alpha = 0.5

    #Run function, don't do anything
    colored_img = color_transfer(style_img, target_img, alpha)
    helper.show_images(
    [style_img, target_img, colored_img], 
    ["Style image", "Target image", "Colored image"], 
    figsize=(15, 8)
    )
run()