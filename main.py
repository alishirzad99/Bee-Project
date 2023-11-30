


import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
import math
# Set paths to your data folders
pollen_folder = 'Data/pollen'
no_pollen_folder = 'Data/no_pollen'

# Get a list of image filenames
bee_images = [os.path.join(pollen_folder, img) for img in os.listdir(pollen_folder)]
non_bee_images = [os.path.join(no_pollen_folder, img) for img in os.listdir(no_pollen_folder)]

# Create labels (1 for bees, 0 for non-bees)
bee_labels = [1] * len(bee_images)
non_bee_labels = [0] * len(non_bee_images)

# Split data into training and testing sets
all_images = bee_images + non_bee_images
all_labels = bee_labels + non_bee_labels
train_images, test_images, train_labels, test_labels = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=42
)
datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# Build the Model
IMG_WIDTH = 180
IMG_HEIGHT = 300
BATCH_SIZE = 100
EPOCHS = 100


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the Models
train_generator = datagen.flow_from_directory(
    'Train_data',
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)
test_generator = datagen.flow_from_directory(
    'Test_data',
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)
# Define your learning rate schedule function
def lr_schedule(epoch):
    initial_lr = 0.001  # Initial learning rate
    drop = 0.5  # Factor by which the learning rate will be reduced
    epochs_drop = 10  # Number of epochs after which the learning rate will be reduced
    lr = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lr


early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(lr_schedule)
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=EPOCHS,
    validation_data=test_generator,
    validation_steps=len(test_generator),
    callbacks=[early_stopping]
)




model.save('primarly_test.h5')
model.save_weights('primarly_wights.h5')

test_loss, test_acc = model.evaluate(test_generator)
print(f'Test Accuracy: {test_acc}')
