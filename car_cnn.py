#imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
import pickle
import requests
from PIL import Image
import cv2
import random

np.random.seed(0)

#import utils
from util_funcs import *
#X_train, y_train, X_test, y_test = None, None, None, None



def main():
    #data loading and preprocessing
    X_train, y_train, X_test, y_test = load_data()




    #data exploration
    explore_data(X_train, y_train, X_test, y_test)

#stage 1 - data preprocessing
def load_data():
    train_data, test_data = unpickle()

    X_train, y_train = train_data["data"], train_data["labels"]
    X_test, y_test = test_data["data"], test_data["labels"]

    #reshape to 32x32x3
    #the transpose line fixes I had with images not being rendered properly (ChatGPT)
    X_train = X_train.reshape(-1, 3, 32, 32)
    X_train = X_train.transpose(0, 2, 3, 1)

    X_test = X_test.reshape(-1, 3, 32, 32)
    X_test = X_test.transpose(0, 2, 3, 1)

    
    #ensure data is of correct form through assert statements
    validate_data(X_train, y_train, X_test, y_test) 

    #check typical image
    display_image(X_train[500], "Typical Image: (Before processing)")

    X_train, X_test = apply_preprocessing(X_train, X_test)
    return X_train, y_train, X_test, y_test

#stage 2 - data exploration
def explore_data(X_train, y_train, X_test, y_test):
    
    #show typical image
    display_image(X_train[500], "Typical Image: (After processing)")

    #image shape
    print("Final Image Shape:", X_train[500].shape )

    #distribution of classes
    all_class_names = cifar100_class_names_coarse + cifar100_class_names_fine + cifar10_class_names
    num_classes = len(all_class_names)
    num_samples = calculate_training_sample_dist(num_classes, all_class_names, X_train, y_train)
    plot_training_sample_dist(num_classes, num_samples)




if __name__ == "__main__":
    main()

    