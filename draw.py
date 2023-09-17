import cv2 as cv
import numpy as np

blank = np.zeros((500, 500, 3), dtype='uint8') # dtype='uint8' is the default for blank images
cv.imshow('Blank', blank)

# 1. Paint the image a certain color
# blank[:] = 0, 255, 0 # green
# cv.imshow('Green', blank)
# blank[200:300, 300:400] = 0, 0, 255 # red
# cv.imshow('Red', blank)

# 2. Draw a rectangle
# cv.rectangle(blank, (0, 0), (blank.shape[1]//2, blank.shape[0]//2), (0, 255, 0), thickness=2) # thickness=-1 or thickness=cv.FILLED will fill the rectangle
# cv.imshow('Rectangle', blank)

# 3. Draw a circle
# cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0, 0, 255), thickness=3) # thickness=-1 or thickness=cv.FILLED will fill the circle
# cv.imshow('Circle', blank)

# 4. Draw a line
# cv.line(blank, (100, 250), (300, 400), (255, 255, 255), thickness=3)
# cv.imshow('Line', blank)

# 5. Write text on an image
cv.putText(blank, 'Hello, my name is Arsalan', (15, 255), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), thickness=2)
cv.imshow('Text', blank)

cv.waitKey(0)
