# CNN-model with tensorflow for image classification for images in ../sourcedata/images_simple/ in 5 classes in 5 folders for labeling

import tensorflow as tf
import numpy as np

# Parameters
learning_rate = 0.001
epochs = 10
batch_size = 16
display_step = 1

# Network Parameters
n_input = 128*128*3
n_classes = 5
dropout = 0.75

# import data with keras from_dir
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255)
train = datagen.flow_from_directory('../sourcedata/images_simple/train', target_size=(128, 128), batch_size=batch_size)
test = datagen.flow_from_directory('../sourcedata/images_simple/test', target_size=(128, 128), batch_size=batch_size)

# define model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

# compile model
model.compile(optimizer= keras.optimizers.Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# train model
model.fit(train, epochs=epochs, validation_data=test)

# save model
model.save('../3_Model/model.h5')
print('Model saved')



