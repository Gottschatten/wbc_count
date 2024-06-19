
import matplotlib.pyplot as plt
import numpy as np
import os
# set logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras

# Params
BATCH_SIZE = 32
EPOCHS = 10
IMG_SIZE = (224, 224)
LEARNING_RATE = 0.01
RETRAIN_LAYER = 35
SEED = 123

#import data 

train_data = keras.preprocessing.image_dataset_from_directory(
    '../1_DatasetCharacteristics/train/',
    validation_split=0.2,
    subset='training',
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_data = keras.preprocessing.image_dataset_from_directory(
    '../1_DatasetCharacteristics/train/',
    validation_split=0.2,
    subset='validation',
    seed=SEED,
    image_size=IMG_SIZE,
)

# normalization layer and scale for ResNet152V2
norm_layer = keras.layers.Rescaling(1/127.5, offset=-1)

# define and use basemodel ResNet 152V2
base_model = keras.applications.ResNet152V2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# freeze the basemodel for it to run in inference mode
base_model.trainable = False

input = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = input
x = keras.applications.resnet_v2.preprocess_input(x)
x = base_model(x, training=False)

# add global average pooling layer
x = keras.layers.GlobalAveragePooling2D()(x)

# add dropout layer
x = keras.layers.Dropout(0.2)(x)

# define output layer
outputs = keras.layers.Dense(4, activation='softmax')(x)

# define model input and output
model = keras.Model(input, outputs)
model.summary()

# compile model
opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train model
history = model.fit(train_data, epochs=EPOCHS, validation_data=test_data, verbose=1)

#unfreeze parts of the basemodel
for layer in base_model.layers[-RETRAIN_LAYER:]:
    if not isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable = True
        
model.summary()

# compile model
opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE/10)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train model
history = model.fit(train_data, epochs=EPOCHS, validation_data=test_data, verbose=1)

#save model
model.save('wbc_count.keras')
print('Model saved')

#save history
np.save('history.npy', history.history)
print('History saved')
# plot training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'], loc='upper left')

# save plot
plt.savefig('accuracy.png')


# compare predictions with actual labels
predictions = model.predict(test_data)
predictions = np.argmax(predictions, axis=1)
actual = np.concatenate([y for x, y in test_data], axis=0)
print(predictions)
print(actual)

# calculate accuracy
accuracy = np.mean(predictions == actual)
print(f'Accuracy: {accuracy}')

#save accuracy to file
with open('model-accuracy.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy}')
