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
from PIL import Image



def get_gradient(img):
    #gets the gradient image


    grad_img = cv2.Laplacian(img,cv2.CV_64F)
    grad_img = np.abs(grad_img)

    #plt.imshow(grad_img, cmap="gray")
    
    return grad_img

def get_edges(img):
    #gets the edges in binary
    
    grad_img = cv2.Canny(img,100,200)

    #grad_img[grad_img == 255] = 1
    #grad_img[grad_img == 0] = 255
    #grad_img[grad_img == 1] = 0
    
    return grad_img





def gradient_prioritize(grad_img, edge_img):
    priority = []
    '''
    #basic top to bottom order
    for i in range(h):
        for j in range(w):
            if edge_img[i,j] == 0:
                priority.append((i,j))
                #s += 1
    '''
    #code for using basic results of contours 
    #Next steps include developing a priority weight based on length of line,
    #proximity to center, strength of gradient, and other factors.
    
    contours = cv2.findContours(edge_img,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = grad_img.shape    
#     print(image)
#     print(hierarchy)
#     pp = pprint.PrettyPrinter(indent=4)
#     pp.pprint(lines.shape)
    

#     for i in lines[1]:
#         for j in i:
#             coord = tuple(map(tuple, j))
#             priority.append(coord[0])
#                 s += 1       
    centerpt = (math.floor(h/2), math.floor(w/2))
    nrmFactor = 255/((h+w)/2)
    longest = len(max(contours[1], key = len))
    lengthNorm = 255/longest
    
    pri_list = []
    
    for j, i in enumerate(contours[1]):
        #find line distance from center, normalize to 256 scale
        avg_pt = np.average(i, axis = 0)
        center_dist = math.ceil(abs(centerpt[0] - avg_pt[0][0]) + abs(centerpt[1] - avg_pt[0][1]))

        center_dist = center_dist * nrmFactor
        center_dist = 255 - center_dist
        
        #find average gradient intensity 
        n = 0
        ints = 0
        for r in i:
            n += 1
            y = r[0][0]
            x = r[0][1]
            ints += grad_img[y, x]
        avg_ints = ints/n
        
        #find line length, normalize to ln scale
        ln_length = len(i) * lengthNorm
        
        pri_score = .3 * avg_ints + .2 * center_dist + .5 * ln_length
        pri_list.append((pri_score, i))
    
    lst2 = sorted(pri_list, key = lambda score: score[0], reverse=True)
    
    priority = []
    filter_contours = []
    average_length = sum(map(len, contours))/float(len(contours))
    img = np.zeros([h, w, 3])
    for i in range(len(contours)):
        cnt = contours[i]
        if (len(cnt) > average_length):
            filter_contours.append(cnt)
            img = cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
            draw_img = cv2.cvtColor(np.rint(img).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            priority.append(draw_img/255)

    for i in range(math.ceil(len(priority)/5) ):
        plot_img_gray(priority[0+i*5:i*5+5],[])
        
    priority_pixel = []
    for i in range(len(filter_contours)):
        for j in range(len(filter_contours[i])):
            priority_pixel.append(filter_contours[i][j][0])
            
    #lines, line_h = cv.findContours(edge_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
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
            x, y = priority[pix]
            sketch_img[y, x] = img[y, x]
            pix += 1
        output_series.append(Image.fromarray(cv2.cvtColor(
           sketch_img, cv2.COLOR_GRAY2RGB)))

        #adjusting last frame so it will finish the remaining pixels
        if i == frames - 2:
            speed = len(priority) - pix - 1
    
    end_merge = sketch_to_original(sketch_img, img)
    
    output_series += end_merge
    
    output_series[0].save(filepath, save_all = True,
                          append_images = output_series[1:], duration = 40)
    
            
al_img = cv2.cvtColor(cv2.imread('images/dog.png'), cv2.COLOR_BGR2RGB)
al_img = cv2.cvtColor(al_img, cv2.COLOR_BGR2GRAY)
#plt.imshow(al_img, cmap="gray")
filepath = 'C:/users/tom/documents/github/cs445-final-project/output/al_sketch.gif'
gif_creator(al_img, 20, filepath)


    