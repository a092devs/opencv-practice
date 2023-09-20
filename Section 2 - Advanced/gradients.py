import cv2 as cv
import numpy as np

img = cv.imread('..Resources/Photos/park.jpg')
cv.imshow('Park', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Laplacian
'''
The Laplacian is a 2-D isotropic measure of the 2nd spatial derivative of an image.
The Laplacian of an image highlights regions of rapid intensity change and is therefore often used for edge detection.
'''
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian', lap)

# Sobel
'''
Sobel operators is a joint Gausssian smoothing plus differentiation operation, so it is more resistant to noise.
You can specify the direction of derivatives to be taken, vertical or horizontal (by the arguments, yorder and xorder respectively).
You can also specify the size of kernel by the argument ksize.
If ksize = -1, a 3x3 Scharr filter is used which gives better results than 3x3 Sobel filter.
'''
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
combined_sobel = cv.bitwise_or(sobelx, sobely)

cv.imshow('Sobel X', sobelx)
cv.imshow('Sobel Y', sobely)
cv.imshow('Combined Sobel', combined_sobel)

canny = cv.Canny(gray, 150, 175)
cv.imshow('Canny', canny)

cv.waitKey(0)