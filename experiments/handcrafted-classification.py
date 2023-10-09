import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    classification_report,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import Bunch

import os
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dropout,
    Dense,
    LeakyReLU,
    BatchNormalization,
)
from pathlib import Path

import cv2
from skimage.io import imread
from skimage.transform import resize
from skimage import color, feature, util
from skimage.feature import hog
from skimage.filters import gabor_kernel, gabor
from skimage.color import rgb2gray
from math import pi
import json
import random

tf.config.list_physical_devices('GPU')

# Function for label encoding
def label_enc(label):
    labels_mapping = {
        'Abyssinian': 0, 'Bengal': 1, 'Birman': 2, 'Bombay': 3, 'British Shorthair': 4,
        'Egyptian Mau': 5, 'american bulldog': 6, 'american pit bull terrier': 7,
        'basset hound': 8, 'beagle': 9, 'boxer': 10, 'chihuahua': 11,
        'english cocker spaniel': 12, 'english setter': 13, 'german shorthaired': 14,
        'great pyrenees': 15
    }
    return labels_mapping.get(label, -1)

# Function for label decoding
def label_dec(enc):
    labels_mapping = {
        0: 'Abyssinian', 1: 'Bengal', 2: 'Birman', 3: 'Bombay', 4: 'British Shorthair',
        5: 'Egyptian Mau', 6: 'american bulldog', 7: 'american pit bull terrier',
        8: 'basset hound', 9: 'beagle', 10: 'boxer', 11: 'chihuahua',
        12: 'english cocker spaniel', 13: 'english setter', 14: 'german shorthaired',
        15: 'great pyrenees'
    }
    return labels_mapping.get(enc, 'Unknown')

# Assign image names to features and labels
BASE_PATH = "/content/drive/Shareddrives/PENGCIT/16-class-images"
image_names = [os.path.basename(file) for file in glob.glob(os.path.join(BASE_PATH, '*.jpg'))]
labels = [' '.join(name.split('_')[:-1]) for name in image_names]

data = []
IMAGE_SIZE = (256, 256)

for name in image_names:
    label = ' '.join(name.split('_')[:-1])
    label_encoded = label_enc(label)
    img = load_img(os.path.join(BASE_PATH, name))
    img = tf.image.resize_with_pad(img_to_array(img, dtype='uint8'), *IMAGE_SIZE).numpy().astype('uint8')
    image = np.array(img)
    data.append([image, label_encoded])

features = []
labels = []
for image, label in data:
    features.append(image)
    labels.append(label)

# Extract histogram and color features
X = []
for img in features:
    img_size = np.shape(img)
    x = img.reshape((img_size[0] * img_size[1] * 3))
    x = x / 255
    x = x.astype(np.uint8)
    x = x.reshape((np.shape(img)[0], np.shape(img)[1], 3))
    new_img = x * img
    hsv = cv2.cvtColor(new_img, cv2.COLOR_RGB2HSV)
    h = hsv[..., 0]
    hist = np.bincount(h.ravel(), minlength=256)
    X.append(hist)

pca = PCA(n_components=0.95)
pca.fit(X)
X = pca.transform(X)
X_train, X_test, y_train, y_test_all = train_test_split(X, labels, test_size=0.3, random_state=109)

param_grid = {'var_smoothing': [1e-11, 1e-10, 1e-9], 'priors': [None]}
clf = GridSearchCV(GaussianNB(), param_grid, refit=True, cv=10, scoring='accuracy')

start_fit = time.time()
clf.fit(X_train, y_train)
stop_fit = time.time()

start_predict = time.time()
y_pred = clf.predict(X_test)
stop_predict = time.time()

accuracy = accuracy_score(y_test_all, y_pred)
f1 = f1_score(y_test_all, y_pred, average='macro')

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"Training Time: {stop_fit - start_fit} seconds")
print(f"Prediction Time: {stop_predict - start_predict} seconds")
print(classification_report(y_test_all, y_pred))

# Now let's create helper functions and organize the code further.
# Function to extract HOG features
def HOG_features(data):
    sample_size = data.shape[0]
    hog_features = []
    for i in range(sample_size):
        image = data[i]
        feature = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3))
        hog_features.append(feature)
    return np.array(hog_features)

# Function to extract LBP features
def LBP_features(data):
    sample_size = data.shape[0]
    lbp_features = []
    for i in range(sample_size):
        image = data[i]
        x = feature.local_binary_pattern(rgb2gray(image), 8, 24, method='uniform')
        lbp_features.append(x)
    return np.array(lbp_features)

# Function to extract GABOR features
def GABOR_features(data):
    sample_size = data.shape[0]
    gabor_features = []
    for i in range(sample_size):
        image = data[i]
        feature = gabor(rgb2gray(image), frequency=0.1, theta=pi/4, 
               sigma_x=3.0, sigma_y=5.0, offset=pi/5, n_stds=5)
        gabor_features.append(feature)
    return np.array(gabor_features)

# Function to extract HSV histogram features
def hsv_histogram(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = hsv[..., 0]
    return np.bincount(h.ravel(), minlength=256)

# Ensure the number of image files and unique labels
print(f"Total number of image files: {len(image_names)}")
print(f"Total number of unique labels: {len(np.unique(labels))}")

# Helper function to assign features and labels
def assign_features_and_labels(image_names, BASE_PATH, IMAGE_SIZE):
    data_1 = []
    data_2 = []
    data_3 = []
    data_4 = []
    data_5 = []
    data_6 = []
    data_7 = []
    data_8 = []
    data_all = []

    for name in tqdm(image_names):
        label = ' '.join(name.split('_')[:-1:])
        labelEncoded = label_enc(label)
        
        img = load_img(os.path.join(BASE_PATH, name))
        img = tf.image.resize_with_pad(img_to_array(img, dtype='uint8'), *IMAGE_SIZE).numpy().astype('uint8')
        image = np.array(img)

        if labelEncoded in (0, 1):
            data_1.append([image, labelEncoded])
        elif labelEncoded in (2, 3):
            data_2.append([image, labelEncoded])
        elif labelEncoded in (4, 5):
            data_3.append([image, labelEncoded])
        elif labelEncoded in (6, 7):
            data_4.append([image, labelEncoded])
        elif labelEncoded in (8, 9):
            data_5.append([image, labelEncoded])
        elif labelEncoded in (10, 11):
            data_6.append([image, labelEncoded])
        elif labelEncoded in (12, 13):
            data_7.append([image, labelEncoded])
        elif labelEncoded in (14, 15):
            data_8.append([image, labelEncoded])
        
        data_all.append([image, labelEncoded])

    features_1 = [image for image, _ in data_1]
    labels_1 = [label for _, label in data_1]

    features_2 = [image for image, _ in data_2]
    labels_2 = [label for _, label in data_2]

    features_3 = [image for image, _ in data_3]
    labels_3 = [label for _, label in data_3]

    features_4 = [image for image, _ in data_4]
    labels_4 = [label for _, label in data_4]

    features_5 = [image for image, _ in data_5]
    labels_5 = [label for _, label in data_5]

    features_6 = [image for image, _ in data_6]
    labels_6 = [label for _, label in data_6]

    features_7 = [image for image, _ in data_7]
    labels_7 = [label for _, label in data_7]

    features_8 = [image for image, _ in data_8]
    labels_8 = [label for _, label in data_8]

    return (features_1, labels_1, features_2, labels_2, features_3, labels_3, features_4, labels_4,
            features_5, labels_5, features_6, labels_6, features_7, labels_7, features_8, labels_8, data_all)

# Call the function to assign features and labels
(features_1, labels_1, features_2, labels_2, features_3, labels_3, features_4, labels_4,
 features_5, labels_5, features_6, labels_6, features_7, labels_7, features_8, labels_8, data_all) = assign_features_and_labels(image_names, BASE_PATH, IMAGE_SIZE)

# Helper function to predict accuracy for each class
def predict_each_class(features, label, method="hog", pca_component=None):
    if method == "hog":
        X_mod = HOG_features(np.array(features))

        if pca_component is not None:
            pca = PCA(n_components=pca_component)
            X_mod = pca.fit_transform(X_mod)

        X_train, X_test, y_train, y_test = train_test_split(X_mod, label, test_size=0.3, random_state=109)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    elif method == "lbp":
        X_mod = LBP_features(np.array(features))

        if pca_component is not None:
            pca = PCA(n_components=pca_component)
            X_mod = pca.fit_transform(X_mod)

        X_train, X_test, y_train, y_test = train_test_split(X_mod, label, test_size=0.3, random_state=109)
        nsamples_x, nx_x, ny_x = X_train.shape
        X_train = X_train.reshape(nsamples_x, nx_x * ny_x)
        nsamples_x, nx_x, ny_x = X_test.shape
        X_test = X_test.reshape(nsamples_x, nx_x * ny_x)

    elif method == "histogram":
        X_mod = []
        for img in features:
            img_size = np.shape(img)
            x = img.reshape((img_size[0] * img_size[1] * 3))
            x = x / 255
            x = x.astype(np.uint8)
            x = x.reshape((np.shape(img)[0], np.shape(img)[1], 3))
            new_img = x * img
            hist = hsv_histogram(new_img)
            X_mod.append(hist)
        X_mod = np.array(X_mod)

        if pca_component is not None:
            pca = PCA(n_components=pca_component)
            X_mod = pca.fit_transform(X_mod)

        X_train, X_test, y_train, y_test = train_test_split(X_mod, label, test_size=0.3, random_state=109)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    param_grid = {'var_smoothing': [1e-11, 1e-10, 1e-9], 'priors': [None]}
    clf = GridSearchCV(GaussianNB(), param_grid, refit=True, cv=5, scoring='accuracy')

    start_fit = time.time()

    clf.fit(X_train, y_train)

    stop_fit = time.time()
    duration_fit = stop_fit - start_fit

    start_predict = time.time()

    y_pred = clf.predict(X_test)

    stop_predict = time.time()
    duration_predict = stop_predict - start_predict

    print(f"Accuracy: {str(accuracy_score(y_test, y_pred))}")
    print(f"F1 score: {str(f1_score(y_test, y_pred, average='macro'))}")
    print(f"Training time: {duration_fit} seconds")
    print(f"Prediction time: {duration_predict} seconds")
    print('\n')

    return clf, y_test, y_pred, accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro'), duration_fit, duration_predict

# Calculate all classifiers, y_test, y_pred, and accuracy
clf_1, y_test_1, y_pred_1, acc_1, f1_1, speed_fit_1, speed_predict_1 = predict_each_class(features_1, labels_1, "hog", 0.95)
clf_2, y_test_2, y_pred_2, acc_2, f1_2, speed_fit_2, speed_predict_2 = predict_each_class(features_2, labels_2, "hog", 0.95)
clf_3, y_test_3, y_pred_3, acc_3, f1_3, speed_fit_3, speed_predict_3 = predict_each_class(features_3, labels_3, "hog", 0.95)
clf_4, y_test_4, y_pred_4, acc_4, f1_4, speed_fit_4, speed_predict_4 = predict_each_class(features_4, labels_4, "hog", 0.95)
clf_5, y_test_5, y_pred_5, acc_5, f1_5, speed_fit_5, speed_predict_5 = predict_each_class(features_5, labels_5, "hog", 0.95)
clf_6, y_test_6, y_pred_6, acc_6, f1_6, speed_fit_6, speed_predict_6 = predict_each_class(features_6, labels_6, "hog", 0.95)
clf_7, y_test_7, y_pred_7, acc_7, f1_7, speed_fit_7, speed_predict_7 = predict_each_class(features_7, labels_7, "hog", 0.95)
clf_8, y_test_8, y_pred_8, acc_8, f1_8, speed_fit_8, speed_predict_8 = predict_each_class(features_8, labels_8, "hog", 0.95)

# Stacking models for ensemble learning
f = [y_pred_1, y_pred_2, y_pred_3, y_pred_4, y_pred_5,
     y_pred_6, y_pred_7, y_pred_8]
f = np.transpose(f)
y_list = [y_test_1, y_test_2, y_test_3, y_test_4, y_test_5,
          y_test_6, y_test_7, y_test_8]

accuracy = []
skor_f1 = []
speed_fit = []
speed_predict = []

for y in y_list:
    lr = LogisticRegression()

    start_fit = time.time()
    lr.fit(f, y)
    stop_fit = time.time()
    duration_fit = stop_fit - start_fit

    start_pred = time.time()
    pred = lr.predict(f)
    stop_pred = time.time()
    duration_pred = stop_pred - start_pred

    accuracy.append(accuracy_score(y, pred))
    skor_f1.append(f1_score(y, pred, average='macro'))
    speed_fit.append(duration_fit)
    speed_predict.append(duration_pred)
    print(classification_report(y, pred))

# Testing stacking results for per-class labels
print(f"Accuracy: {np.mean(accuracy)}")
print(f"F1 score: {np.mean(skor_f1)}")

speed_fit_each_model = speed_fit_1 + speed_fit_2 + speed_fit_3 + speed_fit_4 + speed_fit_5 +\
  speed_fit_6 + speed_fit_7 + speed_fit_8
speed_last_model_fit = np.sum(speed_fit)
total_speed_fit = speed_fit_each_model + speed_last_model_fit
print(f"Training time: {total_speed_fit} seconds")

speed_predict_each_model = speed_predict_1 + speed_predict_2 + speed_predict_3 + speed_predict_4 + speed_predict_5 +\
  speed_predict_6 + speed_predict_7 + speed_predict_8
speed_last_model_predict = np.sum(speed_predict)
total_speed_predict = speed_predict_each_model + speed_last_model_predict
print(f"Prediction time: {total_speed_predict} seconds")

# Testing stacking results for random labels
lr = LogisticRegression()
y = random.choices(y_test_all, k=120)

start_fit = time.time()
lr.fit(f, y)
stop_fit = time.time()

start_pred = time.time()
y_pred = lr.predict(f)
stop_pred = time.time()

print(f"Accuracy: {accuracy_score(y, y_pred)}")
print(f"F1 score: {f1_score(y, y_pred, average='macro')}")
print(f"Training time: {speed_fit_each_model + (stop_fit - start_fit)} seconds")
print(f"Prediction time: {speed_predict_each_model + (stop_pred - start_pred)} seconds")
print(classification_report(y, y_pred))

# Conclusion of experiments

# Experiment 1: Direct classification into 16 classes
# In the first experiment, we attempted direct classification into all 16 classes.
# We extracted histogram and color features, but the model's accuracy was extremely low.
# This led us to consider implementing ensemble learning.

# Experiment 2: One-vs-One classification followed by stacking
# In the second experiment, we adopted a one-vs-one (OVO) approach for classification,
# where each class is paired against another for binary classification.
# We then stacked the results to implement ensemble learning.

# In this second experiment:
# - We performed feature extraction using histogram, color features, HOG, and LBP.
# - The results improved compared to the direct 16-class classification, although not as good as with ANN.
# - We partitioned the dataset into eight sets (16 divided by 2) for OVO classification.
# - The reason for this partitioning is to facilitate OVO classification, which is better suited for the characteristics of handcrafted features.
# - Handcrafted features seem insufficient for direct classification into 16 classes due to the similarity between images.
# - By stacking the models from the eight classes, we aimed to improve the classification performance.

# Overall, the experiments showed that the handcrafted features are not ideal for directly classifying images with high similarity.
# Implementing OVO classification followed by stacking proved to be a better approach to overcome this challenge.