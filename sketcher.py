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



def get_gradient(img):
    #gets the gradient image


    grad_img = cv2.Laplacian(img,cv2.CV_64F)
    grad_img = np.abs(grad_img)

    #plt.imshow(grad_img, cmap="gray")
    
    return grad_img

def get_edges(img):
    #gets the edges in binary
    
    grad_img = cv2.Canny(img,100,200)

    grad_img[grad_img == 255] = 1
    grad_img[grad_img == 0] = 255
    grad_img[grad_img == 1] = 0
    
    return grad_img

def gradient_prioritize(grad_img, edge_img):
    
    #for now, just labeling each pixel sequentially. Add better criteria later.
    #look into the Canny edge detector algorithm for blob/length detection
    h, w = edge_img.shape
    
    #s = 0
    
    priority = []
    
    for i in range(h):
        for j in range(w):
            if edge_img[i,j] == 0:
                priority.append((i,j))
                #s += 1
    
    #priority = priority[:s]
    
    return priority

def sketch_to_original(draw_img, img):
    #creates a series of images merging between gradient image and original
    draw_img = draw_img.astype(float)
    nf = 60
    img_diff = (img - draw_img)
    img_diff /= nf
    h, w = draw_img.shape
    
    output_series = []
    
    for i in range(nf):
        draw_img += img_diff
        output_series.append(Image.fromarray(cv2.cvtColor(
                np.rint(draw_img).astype(np.uint8), cv2.COLOR_GRAY2RGB)))

    return output_series        
    

def gif_creator(img, speed, filepath):
    #creates gif of sketching, with pixes added at rate of speed parameter
    
    grad_img = get_gradient(img)
    edge_img = get_edges(img)
    priority = gradient_prioritize(grad_img, edge_img)
    
    sketch_img = np.zeros(grad_img.shape).astype(np.uint8)
    sketch_img += 255
    
    h, w = grad_img.shape
    pix = 0

    frames = math.ceil(len(priority)/speed)
    
    output_series = []
    
    for i in range(frames):
        for j in range(speed):
            y, x = priority[pix]
            sketch_img[y, x] = img[y, x]
            pix += 1
        output_series.append(Image.fromarray(cv2.cvtColor(
           sketch_img, cv2.COLOR_GRAY2RGB)))

        if i == frames - 2:
            speed = len(priority) - pix - 1
    
    end_merge = sketch_to_original(sketch_img, img)
    
    output_series += end_merge
    
    output_series[0].save(filepath, save_all = True,
                          append_images = output_series[1:], duration = 40)
    
            
al_img = cv2.cvtColor(cv2.imread('images/alfred.png'), cv2.COLOR_BGR2RGB)
al_img = cv2.cvtColor(al_img, cv2.COLOR_BGR2GRAY)
#plt.imshow(al_img, cmap="gray")
filepath = 'C:/users/tom/documents/github/cs445-final-project/output/al_sketch.gif'
gif_creator(al_img, 20, filepath)
    
    

    