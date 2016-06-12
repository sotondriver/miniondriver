# -*- coding: utf-8 -*-
"""
Created on 16/6/6 15:35 2016

@author: harry sun
"""
import numpy as np
import pandas as pd
from python_code.preprocess.create_training_data import get_train_data_array


def construct_data_for_lstm(data):
    data1 = data.tolist()
    _data = []
    _label = []
    for idx in range(data.shape[0]/4):
        matrix = []
        temp_idx = idx*4
        matrix.append(data1[temp_idx])
        matrix.append(data1[temp_idx+1])
        matrix.append(data1[temp_idx+2])
        _label.append(data1[temp_idx+3][2])
        _data.append(matrix)

    return _data, _label

def clean_zeros(x, y):
    non_zero_idx = np.where(y > 0)[0]
    size = len(non_zero_idx)
    x_out = x[non_zero_idx]
    y_out = y[non_zero_idx]
    return x_out, y_out, size

def train_data_split(train_list, label_list, train_size=0.7, test_size=0.1):
    """
    This just splits data to training and testing parts
    """
    data_ = range(len(train_list))
    label_ = range(len(train_list))
    shuffle_idx = range(len(train_list))
    # np.random.seed(641)
    np.random.shuffle(shuffle_idx)
    for i in range(len(shuffle_idx)):
        idx_ = shuffle_idx[i]
        data_[i] = train_list[idx_]
        label_[i] = label_list[idx_]

    idx1 = int(round(len(train_list) * train_size))
    idx2 = int(round(len(train_list) * (1 - test_size)))

    train_data = pd.DataFrame(data_)
    label_data = pd.DataFrame(label_)

    x_train = train_data.iloc[0:idx1].as_matrix()
    y_train = label_data.iloc[0:idx1].values
    x_validate = train_data.iloc[idx1:idx2].values
    y_validate = label_data.iloc[idx1:idx2].values
    x_test = train_data.iloc[idx2:].values
    y_test = label_data.iloc[idx2:].values

    return (x_train, y_train), (x_validate, y_validate), (x_test, y_test)


def load_test_data(path):
    temp_test_xdata = pd.read_csv(path + 'test_data.csv')
    temp_test_ydata = pd.read_csv(path + 'test_label.csv')
    test_data = _construct_xdata(temp_test_xdata)
    test_label = _construct_ydata(temp_test_ydata)
    return test_data, test_label

if __name__ == '__main__':
    train_data = get_train_data_array(65)
    data, label = construct_data_for_lstm(train_data)
    train_data_split(data, label)
