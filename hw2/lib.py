import numpy as np
import math
import cv2

def contrast_stretch(img):
    M,N,C=img.shape
    min_val=256
    max_val=0
    mins=[]
    maxs=[]
    for c in range(C):
        for i in range(M):
            if min(img[i,:,c])<min_val:
                min_val=min(img[i,:,c])
            if max(img[i,:,c])>max_val:
                max_val=max(img[i,:,c])
        mins.append(min_val)
        maxs.append(max_val)

    for c in range(C):
        for i in range(M):
            for j in range(N):
                img[i,j,c]=(img[i,j,c]-mins[c])/(maxs[c]-mins[c])*255
    
    return img

def power_law_transformation(img,gamma):
    return np.array(255*(img/255)**gamma,dtype='uint8')

def gaussian_smoothing_filter(size,sigma):
    smooth_filter=np.zeros((size,size))
    size=int(size/2)
    i=0
    for s in range(-size,size+1,1):
        j=0
        for t in range(-size,size+1,1):
            smooth_filter[i,j]=1/(2*math.pi*sigma**2)*math.exp(-((s**2+t**2)/2*sigma**2))
            j+=1
        i+=1
    smooth_filter/=smooth_filter.sum()
    return smooth_filter

def convolution(img,kernel,pad=1):
    kernel_size=kernel.shape[0]
    img_padded=np.zeros((img.shape[0]+2*pad,img.shape[1]+2*pad))
    img_padded[1:-pad,1:-pad]=img
    out=np.zeros((img.shape[0],img.shape[1]))
    # for x in range(img.shape[1]):
    #     for y in range(img.shape[0]):
    #         out[y, x] = (kernel * img_padded[y:y + 3, x:x + 3]).sum()
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            out[x, y] = (kernel * img_padded[x:x + kernel_size, y:y + kernel_size]).sum()
    return out