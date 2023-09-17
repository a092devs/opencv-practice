import cv2 as cv
import numpy as np

img = cv.imread('Photos/cats.jpg')
cv.imshow('Park', img)

blank = np.zeros(img.shape, dtype='uint8')
cv.imshow('Blank', blank)

gray =cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
# cv.imshow('Blur', blur)

# # Canney Edge Detection
# canny = cv.Canny(blur, 125, 175)
# cv.imshow('Canny Edges', canny)

# Thresholding
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY) # 125 --> threshold value, 255 --> max value
cv.imshow('Thresh', thresh)

# Contours are the boundaries of objects
# contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
'''
cv.RETR_LIST --> all contours
cv.RETR_EXTERNAL --> only external contours
cv.RETR_TREE --> all contours in a hierarchy
cv.CHAIN_APPROX_NONE --> all contours
cv.CHAIN_APPROX_SIMPLE --> removes all redundant points and compresses the contour
cv.CHAIN_APPROX_TC89_L1 --> applies Teh-Chin chain approximation algorithm based on l1 norm
cv.CHAIN_APPROX_TC89_KCOS --> applies Teh-Chin chain approximation algorithm based on kcosine algorithm
'''
print(f'{len(contours)} contour(s) found!')

cv.drawContours(blank, contours, -1, (0, 0, 255), 1)
'''
-1 --> draw all contours
0, 0, 255 --> color of the contour
1 --> thickness of the contour
'''

cv.drawContours(img, contours, -1, (0, 0, 255), 1)
cv.imshow('Contours Drawn', blank)

cv.waitKey(0)