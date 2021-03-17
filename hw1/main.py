import cv2
import numpy as np
from lib import contrast_stretch,power_law_transformation,hisogram_equaliztion,gaussian_smoothing_filter,convolution,median_filter,bilateralfilter

for i in range(3,4,1):
    origin=cv2.imread('p1im'+str(i)+'.png')

    # img=contrast_stretch(origin)
    # cv2.imshow('constrat stretching',img)
    # cv2.waitKey(0)

    img=power_law_transformation(origin,3)
    cv2.imshow('power law transformation',img)
    cv2.waitKey(0)

    # r,g,b=cv2.split(origin)
    # cdf_r=hisogram_equaliztion(r)
    # cdf_g=hisogram_equaliztion(g)
    # cdf_b=hisogram_equaliztion(b)
    # equ=cv2.merge((cdf_r[r],cdf_g[g],cdf_b[b]))
    # cv2.imshow('histogream equalization',equ)
    # cv2.waitKey(0)
'''

sigma_s = 2.8
sigma_r = 0.1*255
for i in range(1,7,1):
    origin=cv2.imread('p1im'+str(i)+'.png')

    img=median_filter(origin,3)
    cv2.imshow('meidan filter',img)
    cv2.waitKey(0)

    kernel=gaussian_smoothing_filter(3,1)
    img=convolution(kernel,origin)
    cv2.imshow('gaussian filter',img)
    cv2.waitKey(0)

    img_gray=cv2.cvtColor(origin,cv2.COLOR_BGR2GRAY)
    img = bilateralfilter(origin,img_gray,sigma_s,sigma_r)
    cv2.imshow('bilateral filter',img)
    cv2.waitKey(0)

'''