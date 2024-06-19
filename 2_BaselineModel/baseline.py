# use keras application resnet152v2 as baseline model with new top of 4 nodes

BATCHSIZE = 32
EPOCHS = 1


from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import preprocessing
import numpy as np

# use keras.preprocessing.image_dataset_from_directory to load images from ./TRAIN, split 80/20 for testing
train_data = preprocessing.image_dataset_from_directory(
    '../1_DatasetCharacteristics/train/',
    validation_split=0.2,
    subset='training',
    seed=123,
)

test_data = preprocessing.image_dataset_from_directory(
    '../1_DatasetCharacteristics/train/',
    validation_split=0.2,
    subset='validation',
    seed=123
)




# load the ResNet152V2 model
base_model = ResNet152V2(weights='imagenet', include_top=False)

# add new top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(4, activation='softmax')(x)

# create the new model
model = Model(inputs=base_model.input, outputs=predictions)

# freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

# print model summary
model.summary()

# train the model
model.fit(train_data, epochs=EPOCHS, batch_size = BATCHSIZE)

# evaluate the model and print the results
model.evaluate(test_data)

# make predictions on test data
predictions = model.predict(test_data)


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
with open('baseline-accuracy.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy}')
