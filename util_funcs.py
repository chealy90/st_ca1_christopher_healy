import pickle
import cv2
import numpy as np
import ast

def unpickle():
    '''
    classes to filter by
    '''
    cifar10_class_list = ["automobile", "bird", "cat", "deer", "dog", "horse","truck"]
    cifar100_class_list_fine = ["cattle", "fox", "baby", "boy",
        "girl", "man", "woman", "rabbit", "squirrel", "bicycle", "bus", "motorcycle", "pickup truck", "train", "lawn-mower", "tractor"]
    cifar100_class_list_coarse = ["trees"]


    '''
    return arrays
    '''
    #init return arrays
    training_batch_names = [f"data_batch_{i}" for i in range(1,6)]    
    training_data = {"data": [], "labels": []} 
    test_data = {"data": [], "labels": []}

    
    '''
    Cifar10
    ''' 

    #training batches   
    for batch in training_batch_names:
        with open(f'cifar10/{batch}', 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            mask = np.isin(data[b"labels"], cifar10_class_list)

            filtered_training_data = np.array(data[b"data"])[mask]
            filtered_training_labels = np.array(data[b"labels"])[mask]

            training_data["data"].append(filtered_training_data)
            training_data["labels"].extend(filtered_training_labels)

    #training_data["data"] = np.concatenate(training_data["data"])
    #training_data["labels"] = np.array(training_data["labels"])


        
    #test batch
    with open('cifar10/test_batch', 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        mask = np.isin(data[b"labels"], cifar10_class_list)
        
        filtered_test_data = np.array(data[b"data"])[mask]
        filtered_test_labels = np.array(data[b"labels"])[mask]

        test_data["data"].append(filtered_test_data)
        test_data["labels"].extend(filtered_test_labels)

    print(training_data["labels"])

    '''
    Cifar100
    '''
    #training data
    with open('cifar100/train', 'rb') as f:
        data = pickle.load(f, encoding='bytes')

        fine_labels = np.array(data[b'fine_labels'])
        coarse_labels = np.array(data[b'coarse_labels'])

        mask = np.isin(fine_labels, cifar100_class_list_fine) | np.isin(coarse_labels, cifar100_class_list_coarse)

        filtered_training_data = np.array(data[b'data'])[mask]

        #Label where statement aided by ChatGPT (see chatlog)
        filtered_training_labels = np.where(
            np.isin(fine_labels[mask], cifar100_class_list_fine),
            fine_labels[mask],
            coarse_labels[mask]
        )

        training_data["data"].append(filtered_training_data)
        training_data["labels"].extend(filtered_training_labels)

        
    #test data
    with open('cifar100/test', 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        
        fine_labels = np.array(data[b'fine_labels'])
        coarse_labels = np.array(data[b'coarse_labels'])

        mask = np.isin(fine_labels, cifar100_class_list_fine) | np.isin(coarse_labels, cifar100_class_list_coarse)

        filtered_test_data = np.array(data[b'data'])[mask]

        #Label where statement aided by ChatGPT (see chatlog)
        filtered_test_labels = np.where(
            np.isin(fine_labels[mask], cifar100_class_list_fine),
            fine_labels[mask],
            coarse_labels[mask]
        )

        test_data["data"].append(filtered_test_data)
        test_data["labels"].extend(filtered_test_labels)

        

    # convert to numpy arrays before returning
    training_data = {
        "data": np.concatenate(training_data["data"], axis=0),
        "labels": np.array(training_data["labels"])
    }

    test_data = {
        "data": np.concatenate(test_data["data"], axis=0),
        "labels": np.array(test_data["labels"])
    }

    print(training_data["labels"])

    return training_data, test_data




#def validate_data(train_data, test_data):
    

def grayscale(img):
    img = cv2.cvtColor, cv2.COLOR_RGB2GRAY
    return img

def equalize(img):
    #this only works for grayscale images
    img = cv2.equalizeHist(img)
    return img
