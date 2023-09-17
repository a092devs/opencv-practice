import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('Photos/cats.jpg')
cv.imshow('Cats', img)

blank = np.zeros(img.shape[:2], dtype='uint8')

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

mask = cv.circle(blank, (img.shape[1]//2 + 45, img.shape[0]//2), 100, 255, -1)

masked = cv.bitwise_and(img, img, mask=mask)
cv.imshow('Mask', masked)

# Grayscale histogram
# gray_hist = cv.calcHist([gray], [0], mask, [256], [0, 256])

# plt.figure() # Create a new figure
# plt.title('Grayscale Histogram') # Set the title
# plt.xlabel('Bins') # Set the x-axis label
# plt.ylabel('# of pixels') # Set the y-axis label
# plt.plot(gray_hist) # Plot the histogram
# plt.xlim([0, 256]) # Set the x limits
# plt.show() # Show the figure

# Color histogram
colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist = cv.calcHist([img], [i], mask, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])

plt.title('Color Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.show()

cv.waitKey(0)