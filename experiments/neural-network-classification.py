import pandas as pd
import matplotlib.image as image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Rescaling, Dense, Flatten, RandomFlip, RandomRotation, Dropout, GlobalAveragePooling2D,
    Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, ZeroPadding2D,
    Activation, Add, AveragePooling2D
)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.metrics import classification_report

# Function to encode labels
def label_enc(label):
    if label == 'Abyssinian':
        return 0
    elif label == 'Bengal':
        return 1
    elif label == 'Birman':
        return 2
    elif label == 'Bombay':
        return 3
    elif label == 'British Shorthair':
        return 4
    elif label == 'Egyptian Mau':
        return 5
    elif label == 'american bulldog':
        return 6
    elif label == 'american pit bull terrier':
        return 7
    elif label == 'basset hound':
        return 8
    elif label == 'beagle':
        return 9
    elif label == 'boxer':
        return 10
    elif label == 'chihuahua':
        return 11
    elif label == 'english cocker spaniel':
        return 12
    elif label == 'english setter':
        return 13
    elif label == 'german shorthaired':
        return 14
    elif label == 'great pyrenees':
        return 15

def show_acc_loss(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

# Constants
BASE_PATH = "/kaggle/input/the-oxfordiiit-pet-dataset/images"
IMAGE_SIZE = (224, 224)

# Function to encode labels
def label_enc(label):
    label_encoder = LabelEncoder()
    label_encoded = label_encoder.fit_transform(label)
    return label_encoded

# Load image names and labels
image_names = [os.path.basename(file) for file in glob.glob(os.path.join(BASE_PATH, '*.jpg'))]
labels = [' '.join(name.split('_')[:-1]) for name in image_names]

# Initialize lists for features and labels
features = []
encoded_labels = []

# Load and preprocess images
for name in image_names:
    label = ' '.join(name.split('_')[:-1])
    label_encoded = label_enc(label)
    if label_encoded is not None:
        img = load_img(os.path.join(BASE_PATH, name))
        img = tf.image.resize_with_pad(img_to_array(img, dtype='uint8'), *IMAGE_SIZE).numpy().astype('uint8')
        image = np.array(img)
        features.append(image)
        encoded_labels.append(label_encoded)

# Convert features and labels to numpy arrays
features_arr = np.array(features)
labels_arr = np.array(encoded_labels)

# One-hot encode labels
labels_one_hot = pd.get_dummies(labels_arr)

# Split dataset into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(features_arr, labels_one_hot, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

# Data augmentation
data_augmentation = Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

# Initialize preprocessing input and prediction layers
preprocess_input = tf.keras.applications.resnet50.preprocess_input
prediction_layer = Dense(16, activation='softmax')

# Define the base model (ResNet50)
resnet_model = ResNet50(include_top=False, pooling='avg', weights='imagenet')
resnet_model.trainable = False

# Build the model
inputs = tf.keras.layers.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = resnet_model(x, training=False)
x = Dropout(0.2)(x)
outputs = prediction_layer(x)
model = Model(inputs, outputs)

# Compile the model
model.compile(
    optimizer=Adam(),
    loss=CategoricalCrossentropy(),
    metrics=['accuracy']
)

# Train the model
model_history = model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=10)

# Show accuracy and loss history
show_acc_loss(model_history, 10)

# Evaluate the model on the test data
model.evaluate(X_test, y_test)

# Make predictions on the test data
y_pred = model.predict(X_test)
print(classification_report(y_test.values, tf.where(y_pred < 0.5, 0, 1)))

# Fine-tuning
print("Number of layers in the base model: ", len(resnet_model.layers))

# Set which layers to fine-tune
resnet_model.trainable = True
fine_tune_at = 120
for layer in resnet_model.layers[:fine_tune_at]:
    layer.trainable = False

print(f"There are {len(model.trainable_variables)} trainable variables")

# Compile the model for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=0.000001),
    loss=CategoricalCrossentropy(),
    metrics=['accuracy']
)

# Fine-tuning training
history_fine = model.fit(x=X_train, y=y_train, 
                         validation_data=(X_val, y_val),
                         epochs=25,
                         initial_epoch=model_history.epoch[-1]
                        )

# Show accuracy and loss history for fine-tuning
show_acc_loss(history_fine, 16)

# Evaluate the fine-tuned model on the test data
model.evaluate(X_test, y_test)

# Make predictions on the test data using the fine-tuned model
y_pred = model.predict(X_test)
print(classification_report(y_test.values, tf.where(y_pred < 0.5, 0, 1)))

# Feature extraction and fine-tuning with MobileNetV2
mobilenet_model = MobileNetV2(
    include_top=False,
    weights='imagenet'
)

mobilenet_model.trainable = False

preprocess_input = mobilenet_preprocess_input
global_average_layer = GlobalAveragePooling2D()
prediction_layer = Dense(16)

inputs = Input(shape=(224,224, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = mobilenet_model(x, training=False)
x = global_average_layer(x)
x = Dropout(0.2)(x)
outputs = prediction_layer(x)
model = Model(inputs, outputs)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

mobilenet_history = model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=50)

show_acc_loss(mobilenet_history, 50)

loss, score = model.evaluate(X_test, y_test)
print("Loss:", loss)
print("Score: ", score)

y_pred = model.predict(X_test)
print(classification_report(y_test.values, tf.where(y_pred < 0.5, 0, 1)))

# Fine-tuning
mobilenet_model.trainable = True
print("Number of layers in the base model: ", len(mobilenet_model.layers))

# Fine-tuning layers up to 100
fine_tune_at = 100
for layer in mobilenet_model.layers[:fine_tune_at]:
    layer.trainable = False

# Viewing trainable variables in MobileNet
len(model.trainable_variables)

model.compile(
    tf.keras.optimizers.RMSprop(learning_rate=0.00001),
    loss=CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Continuing fitting
history_fine = model.fit(x=X_train, y=y_train, 
                         validation_data=(X_val, y_val),
                         epochs=75,
                         initial_epoch=mobilenet_history.epoch[-1]
                        )

show_acc_loss(history_fine, 75)

loss, score = model.evaluate(X_test, y_test)
print("loss:", loss)
print("score: ", score)

y_pred = model.predict(X_test)
print(classification_report(y_test.values, tf.where(y_pred < 0.5, 0, 1)))

# Conclusion
# Based on the results, it can be seen that the ResNet model performs better than MobileNet.
# With 10 epochs, ResNet has already achieved an accuracy above 90%, while MobileNet requires up to 50 epochs.
# In terms of accuracy and speed in training and making predictions, ResNet is better than MobileNet.