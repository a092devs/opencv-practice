import cv2 as cv

img = cv.imread('..Resources/Photos/cat.jpg')
cv.imshow('Cat', img)

def rescaleFrame(frame, scale=0.75):
    # Will work for images, videos and live videos
    width = int(frame.shape[1] * scale) # frame.shape[1] is the width of the frame
    height = int(frame.shape[0] * scale) # frame.shape[0] is the height of the frame
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

resized_image = rescaleFrame(img) # default scale is 0.75
cv.imshow('Cat Resized', resized_image) # we can see that the image is smaller

# cv.waitKey(0)

def changeRes(width, height):
    # Only works for live video
    capture.set(3, width) # 3 is the width id
    capture.set(4, height) # 4 is the height id

# Reading Videos
capture = cv.VideoCapture('..Resources/Videos/dog.mp4')

while True:
    isTrue, frame = capture.read()

    frame_resized = rescaleFrame(frame, scale=0.2)

    cv.imshow('Dog', frame)
    cv.imshow('Dog Resized', frame_resized)

    if cv.waitKey(20) & 0xFF == ord('d'): # 20ms delay between each frame and if we press d, it will break
        break

capture.release
cv.destroyAllWindows()