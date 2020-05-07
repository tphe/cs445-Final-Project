# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 12:02:58 2020

@author: Tom Phelan and Will Wang
"""


# System imports
from os import path
import math
from random import sample 
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
import scipy.sparse.linalg as la
import statistics


# Third-Party Imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

import warnings
warnings.filterwarnings('ignore')

import PIL
from PIL import Image, ImageDraw
import pprint
from scipy.interpolate import splprep, splev



def plot_img(images, titles):
    fig, axs = plt.subplots(nrows = 1, ncols = len(images),figsize=(10,10))
    if len(images) > 1:
        for i, p in enumerate(images):
            axs[i].imshow(p)
            if(titles):
                axs[i].set_title(titles[i])
            axs[i].axis('off')
    else:
            axs.imshow(images[0])
            if(titles):
                axs.set_title(titles[0])
            axs.axis('off')
    plt.show()
    
def plot_img_gray(images, titles):
    fig, axs = plt.subplots(nrows = 1, ncols = len(images),figsize=(10,10))
    if len(images) > 1:
        for i, p in enumerate(images):
            axs[i].imshow(p, cmap="gray")
            if(titles):
                axs[i].set_title(titles[i])
            axs[i].axis('off')
    else:
            axs.imshow(images[0], cmap="gray")
            if(titles):
                axs.set_title(titles[0])
            axs.axis('off')
    plt.show()
    
from scipy.ndimage.filters import gaussian_filter




def get_gradient(img):
    #gets the gradient image


    grad_img = cv2.Laplacian(img,cv2.CV_64F)
    grad_img = np.abs(grad_img)

    #plt.imshow(grad_img, cmap="gray")
    
    return grad_img

def get_edges(img):
    #gets the edges in binary
    
    grad_img = cv2.Canny(img,100,200)


    return grad_img

def Sorting(lst): 
    lst2 = sorted(lst, key=len, reverse = True) 
    return lst2 
      

def gradient_prioritize(grad_img, edge_img):
    
    priority = []

    #code for using basic results of contours 
    #then creating a priority weight based on length of line,
    #proximity to center, strength of gradient, and other factors.
    
    contours = cv2.findContours(edge_img,cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    h, w = grad_img.shape   
    
    img = np.zeros([h, w, 3])
    centerpt = (math.floor(h/2), math.floor(w/2))
    nrmFactor = 255/((h+w)/2)
    longest = len(max(contours[1], key = len))
    lengthNorm = 255/longest
    average_length = sum(map(len, contours[1]))/float(len(contours[1]))
    
    pri_list = []
    dst = []
    priority_pixel = []


   
    for i in contours[1]:
        dst.append(len(i))
        if len(i) > average_length:
            #find line distance from center, normalize to 256 scale
            avg_pt = np.average(i, axis = 0)
            center_dist = math.ceil(abs(centerpt[0] - avg_pt[0][1]) + abs(centerpt[1] - avg_pt[0][0]))

            center_dist = center_dist * nrmFactor
            center_dist = 255 - center_dist

            #find average gradient intensity 
            n = 0
            ints = 0
            for r in i:
                n += 1
                x = r[0][0]
                y = r[0][1]
                ints += grad_img[y, x]
            avg_ints = ints/n

            #find line length, normalize to scale
            ln_length = len(i) * lengthNorm

            pri_score = .3 * avg_ints + .2 * center_dist + .5 * ln_length
            pri_list.append((pri_score, i))
            
            img = cv2.drawContours(img, [i], 0, (0,255,0), 3)
            priority.append(np.copy(img)/255)


    lst2 = sorted(pri_list, key = lambda score: score[0], reverse=True)
    
    plt.hist(dst, bins=100)
    plt.show()
        
    for i in range(math.ceil(len(priority)/5) ):
         plot_img(priority[0+i*5:i*5+5],[])
    
    
    for i in lst2:
        for j in i[1]:
            priority_pixel.append(j[0])
    
    return priority_pixel



def sketch_to_original(draw_img, img):
    #creates a series of images merging between gradient image and original
    draw_img = draw_img.astype(float)
    nf = 30
    img_diff = (img - draw_img)
    img_diff /= nf
    h, w = draw_img.shape
    
    output_series = []
    
    for i in range(nf):
        draw_img += img_diff
        output_series.append(Image.fromarray(cv2.cvtColor(np.rint(draw_img).astype(np.uint8), cv2.COLOR_GRAY2RGB)))

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
            x, y = priority[pix]
            sketch_img[y, x] = 0
            pix += 1
        output_series.append(Image.fromarray(cv2.cvtColor(sketch_img, cv2.COLOR_GRAY2RGB)))

        #adjusting last frame so it will finish the remaining pixels
        if i == frames - 2:
            speed = len(priority) - pix - 1
    
    end_merge = sketch_to_original(sketch_img, img)
    
    output_series += end_merge
    
    output_series[0].save(filepath, save_all = True,
                          append_images = output_series[1:], loop = 0, duration = 60)
    
            
al_img = cv2.cvtColor(cv2.imread('images/naroto.png'), cv2.COLOR_BGR2RGB)
al_img = cv2.cvtColor(al_img, cv2.COLOR_BGR2GRAY)
#plt.imshow(al_img, cmap="gray")
filepath = 'output/naroto.png'
gif_creator(al_img, 80, filepath)


    