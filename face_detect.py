import cv2 as cv

img = cv.imread('Photos/group 1.jpg')
cv.imshow('Group of People', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Person', gray)

# haar cascade
# use local haar cascade file
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# use opencv haar cascade file
# haar_cascade = f'{cv.data.haarcascades}haarcascade_frontalface_default.xml'
# haar_cascade = cv.CascadeClassifier(haar_cascade)

# detect faces
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

print(f'Number of faces found = {len(faces_rect)}')

# draw rectangle around the faces
for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Faces', img)  

cv.waitKey(0)