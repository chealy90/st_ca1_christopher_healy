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
train_data, test_data = None, None
X_train, y_train, X_test, y_test = None, None, None, None


#stage 1 - data preprocessing
def load_data():
    train_data, test_data = unpickle()

    X_train, y_train = train_data["data"], train_data["labels"]
    X_test, y_test = test_data["data"], test_data["labels"]

    #reshape to 32x32x3
    X_train = X_train.reshape(-1, 32, 32, 3)
    X_test = X_test.reshape(-1, 32, 32, 3)

    
    validate_data(X_train, y_train, X_test, y_test) #ensure data is of correct form through assert statements

    X_train, X_test = apply_preprocessing(X_train, X_test)
    

#stage 2 - data exploration





if __name__ == "__main__":
    load_data()

    