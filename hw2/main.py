import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import lib
import math

def plot_threshold_magnitude(thresholds,magnitude,title):
    idx=1
    fig=plt.figure()
    fig.suptitle(title)
    for threshold in thresholds:
        magnitude[magnitude<threshold]=0
        magnitude=magnitude.astype(np.float32)
        plot_id=130+idx
        ax=fig.add_subplot(plot_id)
        idx+=1
        ax.set_title('Threshold '+str(threshold))
        ax.imshow(magnitude, cmap = plt.get_cmap('gray'))

# Normalize the pixel array, so that values are <= 1
def normalize(img):
    img = img/np.max(img)
    return img

def non_max_suppression(mag,gragient_angle):
    nms = np.zeros(mag.shape)
    for i in range(1, int(mag.shape[0]) - 1):
        for j in range(1, int(mag.shape[1]) - 1):
            if((gradient_angle[i,j] >= -22.5 and gradient_angle[i,j] <= 22.5) or (gradient_angle[i,j] <= -157.5 and gradient_angle[i,j] >= 157.5)):
                if((np.alltrue(mag[i,j] > mag[i,j+1])) and (np.alltrue(mag[i,j] > mag[i,j-1]))):
                    nms[i,j] = mag[i,j]
                else:
                    nms[i,j] = 0
            if((gradient_angle[i,j] >= 22.5 and gradient_angle[i,j] <= 67.5) or (gradient_angle[i,j] <= -112.5 and gradient_angle[i,j] >= -157.5)):
                if((mag[i,j] > mag[i+1,j+1]) and (mag[i,j] > mag[i-1,j-1])):
                    nms[i,j] = mag[i,j]
                else:
                    nms[i,j] = 0
            if((gradient_angle[i,j] >= 67.5 and gradient_angle[i,j] <= 112.5) or (gradient_angle[i,j] <= -67.5 and gradient_angle[i,j] >= -112.5)):
                if((mag[i,j] > mag[i+1,j]) and (mag[i,j] > mag[i-1,j])):
                    nms[i,j] = mag[i,j]
                else:
                    nms[i,j] = 0
            if((gradient_angle[i,j] >= 112.5 and gradient_angle[i,j] <= 157.5) or (gradient_angle[i,j] <= -22.5 and gradient_angle[i,j] >= -67.5)):
                if((mag[i,j] > mag[i+1,j-1]) and (mag[i,j] > mag[i-1,j+1])):
                    nms[i,j] = mag[i,j]
                else:
                    nms[i,j] = 0

    return nms

def double_threshold(img):
    highThresholdRatio = 0.4
    lowThresholdRatio = 0.15
    GSup = np.copy(img)
    h = int(GSup.shape[0])
    w = int(GSup.shape[1])
    highThreshold = np.max(GSup) * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio    
    strong_edge_cnt = 0.1
    old_cnt=0
    
    # The while loop is used so that the loop will keep executing till the number of strong edges do not change, i.e all weak edges connected to strong edges have been found
    while(old_cnt != strong_edge_cnt):
        old_cnt = strong_edge_cnt
        for i in range(1,h-1):
            for j in range(1,w-1):
                if(GSup[i,j] > highThreshold):
                    GSup[i,j] = 1
                elif(GSup[i,j] < lowThreshold):
                    GSup[i,j] = 0
                else:
                    if((GSup[i-1,j-1] > highThreshold) or 
                        (GSup[i-1,j] > highThreshold) or
                        (GSup[i-1,j+1] > highThreshold) or
                        (GSup[i,j-1] > highThreshold) or
                        (GSup[i,j+1] > highThreshold) or
                        (GSup[i+1,j-1] > highThreshold) or
                        (GSup[i+1,j] > highThreshold) or
                        (GSup[i+1,j+1] > highThreshold)):
                        GSup[i,j] = 1
        strong_edge_cnt = np.sum(GSup == 1)
    
    GSup = (GSup == 1) * GSup # remove weak edges
    
    return GSup

def laplacian_of_gaussian(img, sigma=0.8, kappa=0.75, pad=False):
    gaussian_filter=lib.gaussian_smoothing_filter(3,0.8)
    img=lib.convolution(img,gaussian_filter)
	
    kernel=np.array([[0,1,0],[1,-4,1],[0,1,0]])
    img=lib.convolution(img,kernel)

    rows, cols = img.shape[:2]
    min_map = np.minimum.reduce(list(img[r:rows-2+r, c:cols-2+c]
                                     for r in range(3) for c in range(3)))
    max_map = np.maximum.reduce(list(img[r:rows-2+r, c:cols-2+c]
                                     for r in range(3) for c in range(3)))
    pos_img = 0 < img[1:rows-1, 1:cols-1]
    neg_min = min_map < 0
    neg_min[1 - pos_img] = 0
    pos_max = 0 < max_map
    pos_max[pos_img] = 0
    zero_cross = neg_min + pos_max
    value_scale = 255. / max(1., img.max() - img.min())
    values = value_scale * (max_map - min_map)
    values[1 - zero_cross] = 0.
    if 0. <= kappa:
        thresh = float(np.absolute(img).mean()) * kappa
        values[values < thresh] = 0.
    log_img = values.astype(np.uint8)
    if pad:
        log_img = np.pad(log_img, pad_width=1, mode='constant', constant_values=0)
    return log_img

def canny(mag,gradient_angle):
    nms=non_max_suppression(mag,gradient_angle)
    output=double_threshold(nms)
    plt.imshow(output, cmap = plt.get_cmap('gray'))
    plt.show()

thresholds=[0.15,0.3,0.45]
s_kernelx=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
s_kernely=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

p_kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
p_kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])


for filename in os.listdir('input_imgs/'):
    img=cv2.imread('input_imgs/'+filename)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    ''' preprocessing '''
    img=lib.contrast_stretch(img)
    gaussian_filter=lib.gaussian_smoothing_filter(3,0.8)
    img=lib.convolution(img,gaussian_filter)

    
    sobelx=lib.convolution(img,s_kernelx)
    sobely=lib.convolution(img,s_kernely)

    prewittx=lib.convolution(img,p_kernelx)
    prewitty=lib.convolution(img,p_kernely)

    sobelx=normalize(sobelx)
    sobely=normalize(sobely)
    prewittx=normalize(prewittx)
    prewitty=normalize(prewitty)


    s_gradient_magnitude=np.hypot(sobelx,sobely)
    s_gradient_magnitude=normalize(s_gradient_magnitude)
    p_gradient_magnitude=np.hypot(prewittx,prewitty)
    p_gradient_magnitude=normalize(p_gradient_magnitude)


    plot_threshold_magnitude(thresholds,s_gradient_magnitude,'Sobel')
    plot_threshold_magnitude(thresholds,p_gradient_magnitude,'Prewitt')
    

    ''' LoG '''
    log=laplacian_of_gaussian(img)
    plt.imshow(log,cmap = plt.get_cmap('gray'))
    plt.show()

    ''' canny '''
    gradient_angle=np.degrees(np.arctan2(sobely,sobelx))
    canny(s_gradient_magnitude,gradient_angle)

plt.show()
