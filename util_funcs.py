import pickle
import cv2
import numpy as np

def unpickle():
    #init return arrays
    training_batch_names = [f"data_batch_{i}" for i in range(1,6)]    
    training_data = {"data": [], "labels": []} 
    test_data = {"data": [], "labels": []}

    #training batches   
    for batch in training_batch_names:
        with open(f'cifar10/{batch}', 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            training_data["data"].append(data[b"data"])
            training_data["labels"].extend(data[b"labels"])

    training_data["data"] = np.concatenate(training_data["data"])
    training_data["labels"] = np.array(training_data["labels"])

        
    #test batch
    with open('cifar10/test_batch', 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        test_data["data"].append(data[b"data"])
        test_data["labels"].extend(data[b"labels"])

        test_data["data"] = np.array(test_data["data"]).reshape(10000, 3072)
        test_data["labels"] = np.array(test_data["labels"])
   
    return training_data, test_data

#def validate_data(train_data, test_data):
    

def grayscale(img):
    img = cv2.cvtColor, cv2.COLOR_RGB2GRAY
    return img

def equalize(img):
    #this only works for grayscale images
    img = cv2.equalizeHist(img)
    return img
