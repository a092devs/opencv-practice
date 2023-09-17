import cv2 as cv
import numpy as np

img = cv.imread('Photos/cats.jpg')
cv.imshow('Cats', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Simple Thresholding
'''
threshold: the value that we are comparing to the pixel value
thresh: the output image
'''
threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY) 
cv.imshow('Simple Thresholded', thresh)

threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV) 
cv.imshow('Simple Thresholded Inverse', thresh)

# Adaptive Thresholding
'''
cv.ADAPTIVE_THRESH_MEAN_C: the threshold value is the mean of the neighbourhood area minus the constant C
cv.ADAPTIVE_THRESH_GAUSSIAN_C: the threshold value is a gaussian-weighted sum of the neighbourhood values minus the constant C
'''
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 9)
cv.imshow('Adaptive Thresholding', adaptive_thresh)

cv.waitKey(0)