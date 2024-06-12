
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Params
BATCH_SIZE = 32
EPOCHS = 10
IMG_SIZE = (224, 224)
LEARNING_RATE = 0.01
RETRAIN_LAYER = 35

#import data 

train_data = keras.preprocessing.image_dataset_from_directory(
    './train',
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_data = keras.preprocessing.image_dataset_from_directory(
    './train',
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=IMG_SIZE,
)

# normalization layer and scale for ResNet152V2
norm_layer = keras.layers.experimental.preprocessing.Rescaling(1/127.5, offset=-1)

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

count = 0
#unfreeze parts of the basemodel
for layer in base_model.layers[-RETRAIN_LAYER:]:
    if not isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable = True
        count +=1
        
print(count)

model.summary()

# compile model
opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE/10)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train model
history = model.fit(train_data, epochs=EPOCHS, validation_data=test_data, verbose=1)

#save model
model.save('model2.keras')
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
plt.show()