import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy.special import softmax

from sklearn.metrics import multilabel_confusion_matrix

import os

basepath_1 = 'C:\Academic\ECE - 4370 (Machine_Learning)\Assignments\Assignment_4\Celegans_ModelGen/1'
labelled_images = []
for image in os.listdir(basepath_1):
    if os.path.isfile(os.path.join(basepath_1, image)):
        img = mpimg.imread(os.path.join(basepath_1, image))
        # remove line below to insert 101x101 images
        arr_img = img.reshape([10201])
        arr_img = np.hstack((arr_img, 1))
        labelled_images.append(arr_img)
        
basepath_0 = 'C:\Academic\ECE - 4370 (Machine_Learning)\Assignments\Assignment_4\Celegans_ModelGen/0'
for image in os.listdir(basepath_0):
    if os.path.isfile(os.path.join(basepath_0, image)):
        img = mpimg.imread(os.path.join(basepath_0, image))
        arr_img = img.reshape([10201])
        arr_img = np.hstack((arr_img, 0))
        labelled_images.append(arr_img)
np.random.shuffle(labelled_images)
# W is (M+1)xK -> 2x2
W = np.random.randn(10201+1,2)

m = W.shape[0] - 1
k = W.shape[1]
t = np.zeros((1, k))
for image in labelled_images:
    target = np.zeros(k)
    label = image[-1]
    target[int(label)] = 1
    t = np.vstack((t, target))
t = np.delete(t, 0, axis=0)

x = []
for image in labelled_images:
    image = np.delete(image, -1, axis=0)
    x.append(image)

x = np.array(x).reshape(len(x), 10201)
x = np.hstack((x, np.ones((len(x), 1))))

# change these values to change training time
phi = 0.0001
max_iter = 10000
n = x.shape[0] # num samples
m = x.shape[1] # num measurements per sample
k = W.shape[1] # num classes
for _ in range(max_iter):
    # error = -(1 / m) * (np.dot(t.T, np.log(y)) + np.dot((1 - t).T, np.log(1 - y)))
    a = np.dot(x, W)
    y = softmax(a, axis=1)
    for j in range(k):
        sum_error_gradient = 0
        sum_error_gradient = np.dot((y[:,j] - t[:,j]), x)
        error = sum_error_gradient.reshape(10202,1)
        W[:, j:j+1] = W[:, j:j+1] - (phi * error)

y = softmax(np.dot(x, W), axis=1)

y_pred = np.array([9,9])
for preds in y:
    index = np.argmax(preds)
    arr = np.zeros(len(preds))
    arr[index] = 1
    y_pred = np.vstack((y_pred, arr))
y_pred = np.delete(y_pred, 0, axis=0)

cf_matrix = multilabel_confusion_matrix(t, y_pred, labels=[False, True])
print(cf_matrix)

# these 2 should be very similar
print(y_pred)
print(t)

correct = 0
for i, preds in enumerate(y_pred):
    if np.array_equal(preds, t[i]):
        correct += 1

accuracy = (correct / 11000) * 100
print("Accuracy: " + accuracy + "%")

