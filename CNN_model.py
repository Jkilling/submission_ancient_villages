import numpy as np
import shutil
from shutil import copyfile
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from random import randrange
from sklearn.metrics import classification_report, confusion_matrix

# Create subdirectories and remove existing folders
# Delete all files within train, test and validation folders
if os.path.exists('data\\derived_data\\train') and os.path.isdir('data\\derived_data\\train'):
    shutil.rmtree('data\\derived_data\\train')
if os.path.exists('data\\derived_data\\test') and os.path.isdir('data\\derived_data\\test'):
    shutil.rmtree('data\\derived_data\\test')
if os.path.exists('data\\derived_data\\validation') and os.path.isdir('data\\derived_data\\validation'):
    shutil.rmtree('data\\derived_data\\validation')

# Create new folders for training, test and so on
try:
    os.mkdir('data\\derived_data\\train')
    os.mkdir('data\\derived_data\\test')
    os.mkdir('data\\derived_data\\validation')
    os.mkdir('data\\derived_data\\train\\tell')
    os.mkdir('data\\derived_data\\train\\no_tell')
    os.mkdir('data\\derived_data\\test\\tell')
    os.mkdir('data\\derived_data\\test\\no_tell')
    os.mkdir('data\\derived_data\\validation\\tell')
    os.mkdir('data\\derived_data\\validation\\no_tell')
except OSError:
    pass

# Path preparation
TRAINING_DIR = "data\\derived_data\\train"
VALIDATION_DIR = "data\\derived_data\\validation"

# Randomize and pick  given data
# Create two lists with positive and negative images
positive_images = os.listdir('data\\derived_data\\confirmed_sites')
negative_images = os.listdir('data\\derived_data\\negative_examples')
random_images = os.listdir('data\\derived_data\\tiles')  # Doesn't contain any tells or other tiles already classfied

# Shuffle them
random.shuffle(positive_images)
random.shuffle(negative_images)

# Random sites to increase the number of negative sites
random_sample = random.sample(random_images, len(positive_images)-len(negative_images))

for img_path in random_sample:
    copyfile('data\\derived_data\\tiles\\' + img_path, 'data\\derived_data\\negative_examples\\' + img_path)

# Pick images
negative_images = os.listdir('data\\derived_data\\negative_examples')
positive_80 = np.round(len(positive_images) * 0.8).astype(int)  # we round to integer
negative_80 = np.round(len(negative_images) * 0.8).astype(int)
remaining_20_positive = len(positive_images) - positive_80
remaining_20_negative = len(negative_images) - negative_80
print("{} positive and {} negative images, {} in total for training"
      .format(positive_80, negative_80, positive_80 + negative_80))
print("{} positive image {} negative images, so {} in total, for validation"
      .format(remaining_20_positive, remaining_20_negative, remaining_20_positive + remaining_20_negative))

# Split them
train_positive = positive_images[:positive_80]
validation_positive = positive_images[-remaining_20_positive:]
train_negative = negative_images[:negative_80]
validation_negative = negative_images[-remaining_20_negative:]

# Copy them into the corresponding folders
for img_path in train_positive:
    copyfile('data\\derived_data\\confirmed_sites\\' + img_path, TRAINING_DIR + '\\tell\\' + img_path)

for img_path in train_negative:
    copyfile('data\\derived_data\\negative_examples\\' + img_path, TRAINING_DIR + '\\no_tell\\' + img_path)

for img_path in validation_positive:
    copyfile('data\\derived_data\\confirmed_sites\\' + img_path, VALIDATION_DIR + '\\tell\\' + img_path)

for img_path in validation_negative:
    copyfile('data\\derived_data\\negative_examples\\' + img_path, VALIDATION_DIR + '\\no_tell\\' + img_path)

# Develop the actual model
# Using the ImageDataGenerator to multiply again the number of images for training and testing
training_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(73, 73),
    class_mode='categorical',
    batch_size=70)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(73, 73),
    class_mode='categorical',
    batch_size=26)

# Define preliminary model for finding a good Learning Rate
tf.keras.backend.clear_session()
model = tf.keras.models.Sequential([
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (4, 4), activation='relu', input_shape=(73, 73, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (4, 4), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    # 512 neuron hidden layer
    #tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')])  # sigmoid or softmax

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.RMSprop(lr=1e-8)
model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
history = model.fit(train_generator, epochs=100, callbacks=[lr_schedule])

# Plot loss and Learning Rate to find optimal value
plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-8, 1e-2, 0, 1])

# Define the actual model
tf.keras.backend.clear_session()
model = tf.keras.models.Sequential([
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (4, 4), activation='relu', input_shape=(73, 73, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (4, 4), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    #tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    # 512 neuron hidden layer
    #tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    #tf.keras.layers.Dense(128, activation='relu'),
    #tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')])  # sigmoid or softmax

model.summary()

optimizer1 = tf.keras.optimizers.RMSprop(lr=1e-3)
optimizer2 = tf.keras.optimizers.RMSprop(lr=1e-4)
optimizer3 = tf.keras.optimizers.Adam(lr=1e-4)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer3,
              metrics=['accuracy'])  # custom optimizer or rmsprop

history = model.fit(train_generator, epochs=500, steps_per_epoch=6,
                    validation_data=validation_generator, verbose=1,
                    validation_steps=4)

# Plot the model performance
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.plot(epochs, loss, 'g', label='Loss')
plt.plot(epochs, val_loss, 'y', label='Validation loss')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()

# Confusion matrix and classification Report
validation_generator_con = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(73, 73),
    class_mode='categorical',
    batch_size=4,
    shuffle=False)

number_val = 104  # Number of images in validation Tell and No_Tell
batch_size = 4
Y_pred = model.predict_generator(validation_generator_con, number_val / batch_size)
print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = ['No_Tell', 'Tell']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))

# Show images and predictions
validation_generator_img = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(73, 73),
    class_mode='binary',
    batch_size=106,
    shuffle=False)

sample_training_images, labels = next(validation_generator_img)
print(sample_training_images)
print(labels)
n_plot = 106
plt.figure(figsize=(16, 10))
index = -1
for i in range(n_plot):
    # random_index = randrange(len(sample_training_images))
    index = index + 1
    prediction = model.predict(np.expand_dims(sample_training_images[index], 0))
    plt.subplot(11, n_plot/10, i + 1)
    plt.subplots_adjust(hspace=0.5)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.title(str(np.argmax(prediction) == labels[index]))
    plt.imshow(sample_training_images[index])
    print("this time I try with the {}-th image - label is {} - model predicts {}. "
          "Which is label {}, so this prediction is {}".format(index, labels[index], prediction, np.argmax(prediction),
                                                               np.argmax(prediction) == labels[index]))

from PIL import Image
from skimage import transform
def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

image = load('data\\derived_data\\tiles\\tile_26.tif')
image = np.delete(image,0,3)
print(image.shape)
model.predict(image)

path=
