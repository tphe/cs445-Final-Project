# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 12:02:58 2020

@author: Tom Phelan and Will Wang
"""



import cv2
import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
import math
import os
import scipy
import scipy.sparse.linalg
import PIL
from PIL import Image, ImageDraw

al_img = cv2.cvtColor(cv2.imread('images/alfred.png'), cv2.COLOR_BGR2RGB)
al_img = cv2.cvtColor(al_img, cv2.COLOR_BGR2GRAY)
plt.imshow(al_img, cmap="gray")

def get_gradient(img):
    #gets the gradient image
    grad_img = cv2.Canny(img,100,200)

    grad_img[grad_img == 255] = 1
    grad_img[grad_img == 0] = 255
    grad_img[grad_img == 1] = 0

    plt.imshow(grad_img, cmap="gray")
    
    return grad_img

def gradient_prioritize(grad_img):
    
    #for now, just labeling each pixel sequentially. Add better criteria later.
    #look into the Canny edge detector algorithm for blob/length detection
    h, w = grad_img.shape
    
    #s = 0
    
    priority = []
    
    for i in range(h):
        for j in range(w):
            if grad_img[i,j] == 0:
                priority.append((i,j))
                #s += 1
    
    #priority = priority[:s]
    
    return priority

def sketch_to_original(grad_img, img):
    #creates a series of images merging between gradient image and original
    grad_img = grad_img.astype(float)
    nf = 20
    img_diff = (img - grad_img)
    img_diff /= nf
    h, w = grad_img.shape
    
    #output_series = np.zeros((h, w, nf))
    output_series = []
    
    for i in range(nf):
        grad_img += img_diff
        #output_series[:,:,i] = np.rint(grad_img).astype(int)
        output_series.append(Image.fromarray(cv2.cvtColor(
                np.rint(grad_img).astype(np.uint8), cv2.COLOR_GRAY2RGB)))

    return output_series        
    

def gif_creator(grad_img, priority, speed):
    #creates gif of sketching, with pixes added at rate of speed parameter
    sketch_img = np.zeros(grad_img.shape, dtype = np.uint8)
    h, w = grad_img.shape
    pix = 0

    frames = math.ceil(len(priority)/speed)
    
    output_series = []
    
    for i in range(frames):
        for j in range(speed):
            y, x = priority[pix]
            sketch_img[y, x] = grad_img[y, x]
            output_series.append(Image.fromarray(cv2.cvtColor(
                sketch_img, cv2.COLOR_GRAY2RGB)))
            pix += 1
        if i == frames - 2:
            speed = len(priority) - pix - 1
    
    output_series[0].save('C:/users/tom/documents/github/cs445-final-project/output/al_sketch.gif', save_all = True,
                          append_images = output_series[1:], duration = 40)
    
            
    
    
    
    
    
    