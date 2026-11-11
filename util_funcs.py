import pickle
import cv2
import numpy as np
import ast

def unpickle():
    '''
    classes to filter by
    '''
    cifar10_class_names = ["automobile", "bird", "cat", "deer", "dog", "horse","truck"]
    cifar100_class_names_fine = ["cattle", "fox", "baby", "boy",
        "girl", "man", "woman", "rabbit", "squirrel", "bicycle", "bus", "motorcycle", "pickup_truck", "train", "lawn_mower", "tractor"]
    cifar100_class_names_coarse = ["trees"]


    '''
    return arrays
    '''
    #init return arrays
    training_batch_names = [f"data_batch_{i}" for i in range(1,6)]    
    training_data = {"data": [], "labels": []} 
    test_data = {"data": [], "labels": []}

    
    ''' ---------------------------------------------------------
    Cifar10
        --------------------------------------------------------
    ''' 
    cifar10_labels = []
    cifar10_class_list = []
    with open(f'cifar10/batches.meta', 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        cifar10_labels = data[b'label_names']
        cifar10_labels = [label.decode('utf-8') for label in cifar10_labels]


    
    for class_name in cifar10_class_names:
        cifar10_class_list.append(cifar10_labels.index(class_name))


    # CIFAR10 TRAINING ------------------------------------------------------------
    for batch in training_batch_names:
        with open(f'cifar10/{batch}', 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            mask = np.isin(data[b"labels"], cifar10_class_list)

            filtered_training_data = np.array(data[b"data"])[mask]
            filtered_training_labels = np.array(data[b"labels"])[mask]

            #switch back to labels
            np_label_names = np.array(cifar10_labels)
            new_labels = np_label_names[filtered_training_labels]

            training_data["data"].append(filtered_training_data)
            training_data["labels"].extend(new_labels)



        
    # CIFAR10 TEST -----------------------------------------
    with open('cifar10/test_batch', 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        mask = np.isin(data[b"labels"], cifar10_class_list)
        
        filtered_test_data = np.array(data[b"data"])[mask]
        filtered_test_labels = np.array(data[b"labels"])[mask]

        #switch back to labels
        np_label_names = np.array(cifar10_labels)
        new_labels = np_label_names[filtered_test_labels]

        #add to final data set
        test_data["data"].append(filtered_test_data)
        test_data["labels"].extend(new_labels)


    ''' -------------------------------------------------------------
    Cifar100
        -------------------------------------------------------------
    ''' 
    cifar100_labels = []
    cifar10_class_list = []

    cifar100_class_list_fine = []
    cifar100_class_list_coarse = []

    # GET LABELS ------------------------------------------------------
    #labels from meta file
    with open('cifar100/meta', 'rb') as f:
        data = pickle.load(f, encoding='bytes')

        cifar100_labels_fine = [label.decode('utf-8') for label in data[b'fine_label_names']]
        cifar100_labels_coarse = [label.decode('utf-8') for label in data[b'coarse_label_names']]


        for class_name in cifar100_class_names_fine:
            cifar100_class_list_fine.append(cifar100_labels_fine.index(class_name))

        for class_name in cifar100_class_names_coarse:
            cifar100_class_list_coarse.append(cifar100_labels_coarse.index(class_name))


    # TRAINING DATA --------------------------------------------------------
    with open('cifar100/train', 'rb') as f:
        data = pickle.load(f, encoding='bytes')

        fine_labels = np.array(data[b'fine_labels'])
        coarse_labels = np.array(data[b'coarse_labels'])       

        # FINE MATCHES --------------------------------------------------------------------
        mask_fine = np.isin(fine_labels, cifar100_class_list_fine)
        filtered_training_data_fine = np.array(data[b'data'])[mask_fine]
        filtered_training_labels_fine = fine_labels[mask_fine]

        #switch back to labels
        np_label_names_fine = np.array(cifar100_labels_fine)
        new_labels = np_label_names_fine[filtered_training_labels_fine]

        #add to final data set
        training_data["data"].append(filtered_training_data_fine)
        training_data["labels"].extend(new_labels)


        # COARSE MATCHES ------------------------------------------
        mask_coarse = np.isin(coarse_labels, cifar100_class_list_coarse)
        filtered_training_data_coarse = np.array(data[b'data'])[mask_coarse]
        filtered_training_labels_coarse = coarse_labels[mask_coarse]

        #switch back to labels
        np_label_names_coarse = np.array(cifar100_labels_coarse)
        new_labels = np_label_names_coarse[filtered_training_labels_coarse]

        #add to final data set
        training_data["data"].append(filtered_training_data_coarse)
        training_data["labels"].extend(new_labels)



        
    # TEST DATA --------------------------------------------------
    with open('cifar100/test', 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        
        fine_labels = np.array(data[b'fine_labels'])
        coarse_labels = np.array(data[b'coarse_labels'])       

        # FINE MATCHES --------------------------------------------------------------------
        mask_fine = np.isin(fine_labels, cifar100_class_list_fine)
        filtered_test_data_fine = np.array(data[b'data'])[mask_fine]
        filtered_test_labels_fine = fine_labels[mask_fine]

        #switch back to labels
        np_label_names_fine = np.array(cifar100_labels_fine)
        new_labels = np_label_names_fine[filtered_test_labels_fine]


        #add to final data set
        test_data["data"].append(filtered_test_data_fine)
        test_data["labels"].extend(new_labels)


        # COARSE MATCHES ------------------------------------------
        mask_coarse = np.isin(coarse_labels, cifar100_class_list_coarse)
        filtered_test_data_coarse = np.array(data[b'data'])[mask_coarse]
        filtered_test_labels_coarse = coarse_labels[mask_coarse]

        #switch back to labels
        np_label_names_coarse = np.array(cifar100_labels_coarse)
        new_labels = np_label_names_coarse[filtered_test_labels_coarse]

        #add to final data set
        test_data["data"].append(filtered_test_data_coarse)
        test_data["labels"].extend(new_labels)
    
        

    # convert to numpy arrays before returning
    training_data = {
        "data": np.concatenate(training_data["data"], axis=0),
        "labels": np.array(training_data["labels"])
    }

    test_data = {
        "data": np.concatenate(test_data["data"], axis=0),
        "labels": np.array(test_data["labels"])
    }


    return training_data, test_data




def validate_data(X_train, y_train, X_test, y_test):
    #assert data is in correct shape
    print("X_train shape: ", X_train.shape)
    print("y_train shape: ", y_train.shape)
    assert(X_train.shape[0] == y_train.shape[0]), "The number of training images does not match the number of labels"
    assert(X_test.shape[0] == y_test.shape[0]), "The number of test images does not match the number of labels"
    assert(X_train.shape[1:] == (32, 32, 3)), "The dimensions of the training images are not 32 x 32 x 3"
    assert(X_test.shape[1:] == (32, 32, 3)), "The dimensions of the test images are not 32 x 32 x 3"
    print("All assertions passed")


def apply_preprocessing(X_train, X_test):
    X_train = np.array(list(map(preprocess_image, X_train)))
    X_test = np.array(list(map(preprocess_image, X_test)))

    return X_train, X_test

def preprocess_image(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255

    return img
    

def grayscale(img):
    img = cv2.cvtColor, cv2.COLOR_RGB2GRAY
    return img


def equalize(img):
    #this only works for grayscale images
    img = cv2.equalizeHist(img)
    return img

