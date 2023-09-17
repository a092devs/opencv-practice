import os
import cv2 as cv
import numpy as np

# # get the list of names of all the folder in the train directory using os.listdir()
# p = []
# for i in os.listdir('Faces/train'):
#     p.append(i)
# print(p)

# manually create a list of all the images in the folder
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
DIR = 'Faces/train'

haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training done!')
# print(f'Length of the features = {len(features)}')
# print(f'Length of the labels = {len(labels)}')

# Convert the features and labels list to numpy array
features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the Recognizer on the features list and the labels list
face_recognizer.train(features, np.array(labels))

face_recognizer.save('face_trained.yml')

# Save the trained model
np.save('features.npy', features)
np.save('labels.npy', labels)