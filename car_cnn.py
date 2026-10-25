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

#classification options
cifar10_class_list = ["automobile", "bird", "cat", "deer", "dog", "horse","truck"]
cifar100_class_list = ["cattle", "fox", "baby", "boy",
"girl", "man", "woman", "rabbit", "squirrel", "trees", "bicycle", "bus", "motorcycle", "pickup truck", "train", "lawn-mower", "tractor"]


#stage 1
def load_data():
    train_data, test_data = unpickle()
    X_train, y_train = train_data["data"], train_data["labels"]
    X_test, y_test = test_data["data"], test_data["labels"]

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)


if __name__ == "__main__":
    load_data()

    