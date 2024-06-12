# use keras application resnet152v2 as baseline model with new top of 4 nodes

BATCHSIZE = 32
EPOCHS = 1


from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import preprocessing
# use keras.preprocessing.image_dataset_from_directory to load images from ./TRAIN, split 80/20 for testing
train_data = preprocessing.image_dataset_from_directory(
    './TRAIN',
    validation_split=0.2,
    subset='training',
    seed=123,
)

test_data = preprocessing.image_dataset_from_directory(
    './TRAIN',
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
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# print model summary
model.summary()

# train the model
model.fit(train_data, epochs=EPOCHS, batch_size = BATCHSIZE)

# evaluate the model and print the results
model.evaluate(test_data)

# make predictions on test data
predictions = model.predict(test_data)

# print the predicted labels
print(predictions)

# save the model
model.save('baseline_model.h5')

print('Done!')

