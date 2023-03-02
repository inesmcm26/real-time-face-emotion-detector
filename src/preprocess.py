import opendatasets as od
import pandas as pd
import numpy as np
import os
from keras.utils.np_utils import to_categorical
import tensorflow as tf


def download_data(PWD, OUTPUT_FOLDER):
    """
     Download the dataset from kaggle return a pandas dataframe
    """

    # download the dataset using open datasets
    od.download("https://www.kaggle.com/datasets/deadskull7/fer2013", force = True)
    
    # move the dataset into the data folder
    os.rename(PWD + '/src/fer2013/fer2013.csv', PWD + OUTPUT_FOLDER + '/fer2013.csv')

    # remove empty dir
    os.rmdir(PWD + '/src/fer2013/')


def list2pixel(lists):
    """
    Converts a string of pixeis into a list of floats
    """
    pixels = lists.apply(lambda x: [float(i) for i in x.split()])

    return pixels

def custom_train_test_split(data):

    # split into train, val and test, convert to tensor and reshape
    X_train = tf.reshape(tf.convert_to_tensor(np.array(data[data['Usage'] == 'Training']['pixels'].to_list()),
                                              dtype=tf.float32), (-1, 48, 48, 1))
    
    X_val = tf.reshape(tf.convert_to_tensor(np.array(data[data['Usage'] == 'PublicTest']['pixels'].to_list()),
                                              dtype=tf.float32), (-1, 48, 48, 1))
    X_test = tf.reshape(tf.convert_to_tensor(np.array(data[data['Usage'] == 'PrivateTest']['pixels'].to_list()),
                                              dtype=tf.float32), (-1, 48, 48, 1))

    # convert labels to categorical
    y_train = to_categorical(data[data['Usage'] == 'Training']['emotion'].tolist(), num_classes = 7)
    y_val = to_categorical(data[data['Usage'] == 'PublicTest']['emotion'].tolist(), num_classes= 7)
    y_test = to_categorical(data[data['Usage'] == 'PrivateTest']['emotion'].tolist(), num_classes= 7)

    return X_train, X_val, X_test, y_train, y_val, y_test