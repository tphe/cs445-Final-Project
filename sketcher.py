# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 12:02:58 2020

@author: Tom Phelan and Will Wang
"""



import cv2
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
from utils import *
import os
import scipy
import scipy.sparse.linalg

al_img = cv2.cvtColor(cv2.imread('images/alfred.png'), cv2.COLOR_BGR2RGB)
al_img = cv2.cvtColor(al_img, cv2.COLOR_BGR2GRAY).astype('double') / 255.0
plt.imshow(al_img, cmap="gray")

def get_gradient(img):
    #gets the gradient image
    grad_img = cv2.Canny(img,100,200)

    plt.imshow(grad_img, cmap="gray")
    
    return grad_img

def gradient_prioritize(grad_img):
    
    #for now, just labeling each pixel sequentially. Add better criteria later.
    #look into the Canny edge detector algorithm for blob/length detection
    h, w = grad_img.shape
    
    s = 1
    
    priority = np.zeros(grad_img.shape)
    
    for i in range(len(h)):
        for j in range(len(w)):
            if grad_img[h,w] > 0:
                priority[h,w] = s
                s += 1
    
    return priority

def sketch_to_original(grad_img, img):
    #creates gif of merge between gradient image an original

def gif_creator(grad_img, priority, speed):
    #creates gif of sketching, with pixes added at rate of speed parameter