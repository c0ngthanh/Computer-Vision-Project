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
def _2d_gaussian(sigma: float) -> np.ndarray:
    """
    Returns 2D Gaussian filter.
    :param sigma: controls the kernel strength
    """
    ksize = int(np.ceil(sigma)*6+1)
    gaussian_1d = cv2.getGaussianKernel(ksize, sigma)
    
    return gaussian_1d * np.transpose(gaussian_1d)

def _low_pass_filter(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    Returns low-pass filter.
    :param img: image
    :param sigma: controls the kernel strength
    """
    return cv2.filter2D(img, -1, _2d_gaussian(sigma))

def _high_pass_filter(img: np.ndarray, sigma: float) -> np.ndarray:
    """
    Returns high-pass filter.
    :param img: image
    :param sigma: controls the kernel strength
    """
    return img - _low_pass_filter(img, sigma)
def _gaus_pyramid(img: np.ndarray, depth: int, sigma: int) -> List[np.ndarray]:
    """
    Creates Gaussian pyramid for img.
    :param img: image
    :param depth: depth of the pyramid
    :param sigma: controls the kernel strength
    """
    _im = img.copy()
    
    pyramid = []
    for d in range(depth-1):
        _im = _low_pass_filter(_im.copy(), sigma)
        pyramid.append(_im)
        _im = cv2.pyrDown(_im)
        
    return pyramid 

def _lap_pyramid(img: np.ndarray, depth: int, sigma: int) -> List[np.ndarray]:
    """
    Creates Laplacian pyramid for img.
    :param img: image
    :param depth: depth of the pyramid
    :param sigma: controls the kernel strength
    """
    _im = img.copy()
    
    pyramid = []
    for d in range(depth-1):
        lap = _high_pass_filter(_im.copy(), sigma)
        pyramid.append(lap)
        _im = cv2.pyrDown(_im)
        
    return pyramid 

def _blend(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Return single image by blending 2 images with mask.
    :param img1: image 1
    :param img1: image 2
    :param mask: mask
    """
    return img1 * mask + img2 * (1.0 - mask)
def laplacian_blend(
    img1: np.ndarray, 
    img2: np.ndarray, 
    mask: np.ndarray, 
    depth: int, 
    sigma: int
) -> np.ndarray:
    """
    Performs blending using Laplacian pyramid.
    :param img1: original image 1
    :param img1: original image 2
    :param mask: original mask
    :param depth: depth of the pyramid
    :param sigma: controls the kernel strength
    """
    mask_gaus_pyramid = _gaus_pyramid(mask, depth, sigma)
    img1_lap_pyramid, img2_lap_pyramid = _lap_pyramid(img1, depth, sigma), _lap_pyramid(img2, depth, sigma)

    blended = [_blend(obj, bg, mask) for obj, bg, mask in zip(img1_lap_pyramid, img2_lap_pyramid, mask_gaus_pyramid)][::-1]
    
    h, w = blended[0].shape[:2]
    
    img1 = cv2.resize(img1, (h, w))
    img2 = cv2.resize(img2, (h, w))
    mask = cv2.resize(mask, (h, w))

    blanded_img = _blend(img1, img2, mask)
    blanded_img = cv2.resize(blanded_img, blended[0].shape[:2])
    
    imgs = []
    for d in range(0, depth-1):
        gaussian_img = _low_pass_filter(blanded_img.copy(), sigma)
        reconstructed_img = cv2.add(blended[d], gaussian_img)
        
        imgs.append(reconstructed_img)
        blanded_img = cv2.pyrUp(reconstructed_img)
        
    return np.clip(imgs[-1], 0, 1)
def run():
    # Load Image and set up parameters
    img1 = helper.get_image('samples/img9.jpg')
    img2 = helper.get_image('samples/img6.jpg')
    mask = helper.get_image('samples/mask3.jpg', mask=True)

    #Run function, don't do anything
    mask_stack = np.stack((mask.astype(float),)*3, axis=-1)
    lap_blend = laplacian_blend(img1, img2, mask_stack, 5, 25)
    helper.show_images(
    [img2, img1, mask, lap_blend], 
    ["Background image", "Object image", "Mask image", "Blended image"]
    )
run()