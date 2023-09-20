import cv2 as cv

img = cv.imread('..Resources/Photos/park.jpg')

cv.imshow('Boston', img)

# 1. Convert to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# 2. Blur
blur = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

# 3. Edge Cascade
canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny Edges', canny)

# 4. Dilating the image
dilated = cv.dilate(canny, (7, 7), iterations=3)
cv.imshow('Dilated', dilated)

# 5. Eroding
eroded = cv.erode(dilated, (7, 7), iterations=3)
cv.imshow('Eroded', eroded)

# 6. Resize
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC) # INTER_AREA, INTER_LINEAR, INTER_CUBIC (slowest)
cv.imshow('Resized', resized)

# 7. Cropping
cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)

cv.waitKey(0)