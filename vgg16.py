# -*- coding: utf-8 -*-
"""Vgg16.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_BWwRU8InCvqc4lvypjx-JvGoLDnzbRk

# **Vgg16**
"""

from google.colab import drive
drive.mount('/content/drive')

!pip install visualkeras

import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam

# Define image dimensions, batch size, and number of epochs
img_width, img_height = 224, 224
batch_size = 32
epochs = 4

# Define your paths
train_data = "/content/drive/MyDrive/archive (4)/Indian Currency Dataset/train"
test_data = "/content/drive/MyDrive/archive (4)/Indian Currency Dataset/test"

# Create ImageDataGenerators
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load and prepare your data
train_generator = train_datagen.flow_from_directory(
    train_data,
    target_size=(img_height, img_width),
    class_mode='binary',  # binary classification (fake vs. real)
    batch_size=batch_size,
    shuffle=True,
)

test_generator = test_datagen.flow_from_directory(
    test_data,
    target_size=(img_height, img_width),
    class_mode='binary',
    batch_size=batch_size,
    shuffle=False,
)

from visualkeras import layered_view
from tensorflow.keras.applications import VGG16
from IPython.display import Image, display

# Load the VGG16 model
model = VGG16(weights='imagenet')

# Identify convolutional and max pooling layers
conv_layers = [layer.name for layer in model.layers if 'conv' in layer.name]
maxpool_layers = [layer.name for layer in model.layers if 'max_pooling2d' in layer.name]

# Print the names of convolutional layers
print("Convolutional Layers:", conv_layers)
print("MaxPooling Layers:", maxpool_layers)

vis_model = layered_view(model)
# Save the visualization to an image file
vis_model.save('vgg16_layers.png')
display(Image('vgg16_layers.png'))

for layer in model.layers:
    print(layer.name)

# Build VGG16 model
base_model = VGG16(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')

])
for layer in base_model.layers:
    layer.trainable = False

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size
)

# Save the extracted features for later use
model.save("my_model.h5")

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // batch_size)
print(f'Test Accuracy: {test_acc}')

train_loss, train_acc = model.evaluate(train_generator, steps=test_generator.samples // batch_size)
print(f'Train Accuracy: {train_acc}')

# Plot Training and Validation Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot Training and Validation Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
# Make predictions on test data
num_steps = len(test_generator)
y_pred = model.predict_generator(test_generator, steps=num_steps, verbose=1)
y_pred = (y_pred > 0.5)

# Convert the class labels to integers
y_true = np.array(test_generator.classes)

# Display classification report
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))

# Plot the confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

import numpy as np
from keras.preprocessing import image
from keras.models import load_model

# Load the trained model
model = load_model("my_model.h5")

# Define the path to the image you want to predict
image_path = "/content/drive/MyDrive/archive (4)/Indian Currency Dataset/test/real/Copy of Copy of Copy of Copy of test (10).jpg"
# Load and preprocess the image
img = image.load_img(image_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for the batch
img_array /= 255.0  # Rescale to [0, 1]

# Make predictions
predictions = model.predict(img_array)

# Display the prediction result
if predictions[0, 0] > 0.5:
    print("Predicted class: Real")
else:
    print("Predicted class: Fake")