import os, caer, canaro, gc
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.optimizers.legacy import SGD


IMG_SIZE = (80, 80)
channels = 1
char_path = 'simpsons_dataset'

# Create a character dictionary
char_dict = {
    char: len(os.listdir(os.path.join(char_path, char)))
    for char in os.listdir(char_path)
}

# Sort in descending order
char_dict = caer.sort_dict(char_dict, descending=True)
# print(char_dict)

#  Get the first 10 categories with the most number of images
characters = []
count = 0
for i in char_dict:
    characters.append(i[0])
    count += 1
    if count >= 10:
        break
# print(characters)

# Create the training data
train = caer.preprocess_from_dir(char_path, characters, channels=channels, IMG_SIZE=IMG_SIZE, isShuffle=True)

# # Number of training samples
# print(f'{len(train)}')

# Visualize the data
plt.figure(figsize=(30, 30))
plt.imshow(train[0][0], cmap='gray')
plt.show()

# Separate the array and corresponding labels
featureSet, labels = caer.sep_train(train, IMG_SIZE=IMG_SIZE)

# Normalize the featureSet ==> (0,1)
featureSet = caer.normalize(featureSet)
# print(labels)

# Convert numerical labels to binary class vectors
labels = to_categorical(labels, len(characters))

# Create train and validation data
x_train, x_val, y_train, y_val = caer.train_val_split(featureSet, labels, val_ratio=.2)

# Delete variables to free up memory
del train
del featureSet
del labels
gc.collect()

# Batch size for training (typically 32 or 64)
BATCH_SIZE = 32

# Number of training epochs
EPOCHS = 10

# Image data generator
data_gen = canaro.generators.imageDataGenerator() 
train_gen = data_gen.flow(x_train, y_train, batch_size=BATCH_SIZE)

output_dim = 10
w, h = IMG_SIZE[:2]

# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(w, h,channels)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu')) 
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(1024, activation='relu'))

model.add(Dense(output_dim, activation='softmax'))

model.summary()

# Train the model
optimizer = SGD(learning_rate=0.001, decay=1e-7, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
callbacks_list = [LearningRateScheduler(canaro.lr_schedule)]
training = model.fit(train_gen, steps_per_epoch=len(x_train)//BATCH_SIZE, epochs=EPOCHS, validation_data=(x_val, y_val), validation_steps=len(y_train)//BATCH_SIZE, callbacks=callbacks_list)

print(characters)

# Testing the model using test data
test_path = 'simpson_testset\abraham_grampa_simpson_0.jpg'

img = cv.imread(test_path)

plt.imshow(img, cmap='gray')
plt.show()

# Prepare
def prepare(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, IMG_SIZE)
    img = caer.reshape(img, IMG_SIZE, 1)
    return img

predictions = model.predict(prepare(img))

print(characters[np.argmax(predictions[0])])