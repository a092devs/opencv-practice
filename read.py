import cv2 as cv

# Reading Images
img = cv.imread('Photos/cat_large.jpg')

cv.imshow('Cat', img)
cv.waitKey(0) # 0 means infinite time until we press any key

# Reading Videos
capture = cv.VideoCapture('Videos/dog.mp4')

while True:
    isTrue, frame = capture.read()
    cv.imshow('Dog', frame)

    if cv.waitKey(20) & 0xFF == ord('d'): # 20ms delay between each frame and if we press d, it will break
        break

capture.release
cv.destroyAllWindows()