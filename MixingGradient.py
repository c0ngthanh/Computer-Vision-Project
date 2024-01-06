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
def mixed_blend(
    img_s: np.ndarray, 
    mask: np.ndarray, 
    img_t: np.ndarray
) -> np.ndarray:
    """
    Returns a mixed gradient blended image with masked img_s over the img_t.
    :param img_s: the image containing the foreground object
    :param mask: the mask of the foreground object in img_s
    :param img_t: the background image 
    """
    img_s_h, img_s_w = img_s.shape
    
    nnz = (mask>0).sum()
    im2var = -np.ones(mask.shape[0:2], dtype='int32')
    im2var[mask>0] = np.arange(nnz)
    
    ys, xs = np.where(mask==1) 
        
    A = sp.sparse.lil_matrix((4*nnz, nnz))
    b = np.zeros(4*nnz)
    
    e = 0
    for n in range(nnz):
        y, x = ys[n], xs[n]  
        
        for n_y, n_x in helper.neighbours(y, x, img_s_h-1, img_s_w-1):
            ds = img_s[y][x] - img_s[n_y][n_x]
            dt = img_t[y][x] - img_t[n_y][n_x]
            d = ds if abs(ds) > abs(dt) else dt
            
            A[e, im2var[y][x]] = 1
            b[e] = d
            
            if im2var[n_y][n_x] != -1:
                A[e, im2var[n_y][n_x]] = -1
            else:
                b[e] += img_t[n_y][n_x]
            e += 1
    
    A = sp.sparse.csr_matrix(A)
    v = sp.sparse.linalg.lsqr(A, b)[0]
    
    img_t_out = img_t.copy()
    
    for n in range(nnz):
        y, x = ys[n], xs[n]
        img_t_out[y][x] = v[im2var[y][x]]
    
    return np.clip(img_t_out, 0, 1)
def run():
    # Load Image and set up parameters
    bg_img = helper.get_image('samples/img3.jpg')
    obj_img = helper.get_image('samples/img4.jpg')
    mask_img =  helper.get_image('samples/mask2.jpg', mask=True)

    #Run function, don't do anything
    mix_img = np.zeros(bg_img.shape)
    for b in np.arange(3):
        mix_img[:,:,b] = mixed_blend(obj_img[:,:,b], mask_img, bg_img[:,:,b].copy())
    helper.show_images(
    [bg_img, obj_img, mask_img, mix_img], 
    ["Background image", "Object image", "Mask image", "Blended image"]
    )
run()