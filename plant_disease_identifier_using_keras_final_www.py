# -*- coding: utf-8 -*-
"""plant disease identifier using keras final .ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jmPFDS3EZ02gvA2LYOV9eYj_xghRgUZM
"""

import os
os.mkdir('plants')

/content/drive/My Drive/70909_150545_bundle_archive.zip

!unzip -o '/content/drive/My Drive/70909_150545_bundle_archive.zip' -d plants

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten,Dropout
from keras.layers import Dense


classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(64, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(64,(3,3),activation='relu'))
classifier.add(Dropout(rate=0.1))
# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 15, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/content/plants/PlantVillage',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('/content/plants/PlantVillage',
                                           target_size = (64, 64),
                                           batch_size = 32,
                                           class_mode = 'categorical')
batch_size=32
# checkpoint
from keras.callbacks import ModelCheckpoint
weightpath = "gdrive/My Drive/plant_weights_best.hdf5"
checkpoint = ModelCheckpoint(weightpath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
callbacks_list = [checkpoint]



classifier.fit_generator(training_set,steps_per_epoch = 20638//batch_size, epochs = 20, validation_data = test_set, validation_steps = 20638//batch_size)

classifier.save_weights("/content/drive/My Drive/plant_disease_trained_model.h5")
a=training_set.class_indices.keys()
a=list(a)

# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
import keras

# %pylab inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img=mpimg.imread('/content/plants/plantvillage/PlantVillage/Pepper__bell___Bacterial_spot/0022d6b7-d47c-4ee2-ae9a-392a53f48647___JR_B.Spot 8964.JPG')
imgplot = plt.imshow(img)
plt.show()
img = keras.preprocessing.image.load_img(
    "/content/plants/plantvillage/PlantVillage/Pepper__bell___Bacterial_spot/0022d6b7-d47c-4ee2-ae9a-392a53f48647___JR_B.Spot 8964.JPG", target_size=(64, 64, 3))
img_array = keras.preprocessing.image.img_to_array(img)

img_array = tf.expand_dims(img_array, 0)  

predictions = classifier.predict(img_array)
score = predictions[0]
print(score)

reverse_mapping = [' Corn Northern Leaf Blight','Potato Late Blight','Corn Common Rust','Corn Gray Leaf Spot', 'Corn healthy','Potato Early Blight','Potato Healthy','Tomato Early Blight','Tomato Late Blight','Tomato Leaf Mold','Tomato Target Spot','Tomato Yellow Leaf Curl','Tomato Bacterial Spot','Tomato Two Spotted Spider Mite','Tomato healthy','grape black measles','grape black rot','grape healthy', 'grape leaf blight' ]
for i in range(15):
  if prediction[0][i] == 1:
    print(reverse_mapping[i])
    break