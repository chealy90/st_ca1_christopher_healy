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

#import utils and models
from util_funcs import *
from models import *

np.random.seed(0)

cifar10_class_names = ["automobile", "bird", "cat", "deer", "dog", "horse","truck"]
cifar100_class_names_fine = ["cattle", "fox", "baby", "boy",
        "girl", "man", "woman", "rabbit", "squirrel", "bicycle", "bus", "motorcycle", "pickup_truck", "train", "lawn_mower", "tractor"]
cifar100_class_names_coarse = ["trees"]

class_names = cifar10_class_names + cifar100_class_names_fine + cifar100_class_names_coarse





def main():
    #stage 1 - data preprocessing
    X_train, y_train, X_test, y_test = load_data()

    #stage 2 - data exploration
    explore_data(X_train, y_train, X_test, y_test)

    #separate out validation data set
    X_train, y_train, X_val, y_val = create_train_validation_split(X_train, y_train)

    #prepare labels for one hot encoding
    class_to_index = {name: index for index, name in enumerate(class_names)}

    #print(y_train.shape)
    #print(y_train[:5])
    #print(type(y_train[0]))

    y_train = np.array([class_to_index[label] for label in y_train])
    y_val = np.array([class_to_index[label] for label in y_val])
    y_test = np.array([class_to_index[label] for label in y_test])
    y_train, y_val, y_test = one_hot_encode(24, y_train, y_val, y_test)

    #reshape for datagen
    X_train = X_train.reshape(-1, 32, 32, 1)
    X_val = X_val.reshape(-1, 32, 32, 1)
    X_test = X_test.reshape(-1, 32, 32, 1)

    #


    #stage 3 - build models
    #create models
    num_classes = 24
    basic_cnn = create_basic_cnn(num_classes)
    leNet_model = create_leNet_model(num_classes)
    modified_letNet_model = create_modifed_letNet(num_classes)
    final_model = create_final_model(num_classes)

    #stage 4 - test models
    #evaluate performance
    datagen = create_data_generator()

    #evaluate_model(basic_cnn, "Basic CNN", X_train, y_train, X_val, y_val, X_test, y_test, datagen)
    #evaluate_model(leNet_model, "leNet Model", X_train, y_train, X_val, y_val, X_test, y_test, datagen)
    #evaluate_model(modified_letNet_model, "Modified LeNet Model", X_train, y_train, X_val, y_val, X_test, y_test, datagen)
    evaluate_model(final_model, "Final Model", X_train, y_train, X_val, y_val, X_test, y_test, datagen, 30)



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

    