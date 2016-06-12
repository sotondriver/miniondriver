# -*- coding: utf-8 -*-
"""
Created on 16/6/6 15:35 2016

@author: harry sun
"""
import numpy as np
import pandas as pd
from python_code.preprocess.create_training_data import get_train_data_array_db

def get_train_data_array_csv(district_idx):
    path = '../../processed_data/train/district_'+str(district_idx)+'_training_data.csv'
    train_data = pd.read_csv(path).values
    return train_data



def construct_data_for_lstm(data):
    data1 = data.tolist()
    _data = []
    _label = []
    for idx in range(data.shape[0]-3):
        matrix = []
        temp_idx = idx
        matrix.append(data1[temp_idx])
        matrix.append(data1[temp_idx+1])
        matrix.append(data1[temp_idx+2])
        _label.append(data1[temp_idx+3])
        _data.append(matrix)

    return _data, _label

def clean_zeros_or_not(x, y, flag):
    if flag:
        non_zero_idx = np.where(y > 0)[0]
        size = len(non_zero_idx)
        x_out = x[non_zero_idx]
        y_out = y[non_zero_idx]
    else:
        x_out = x
        y_out = y
        size = x_out.shape[0]
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

    x_train = np.asarray(data_[0:idx1])
    y_train = np.asarray(label_[0:idx1])
    x_validate = np.asarray(data_[idx1:idx2])
    y_validate = np.asarray(label_[idx1:idx2])
    x_test = np.asarray(data_[idx2:])
    y_test = np.asarray(label_[idx2:])

    return (x_train, y_train), (x_validate, y_validate), (x_test, y_test)


def load_test_data(path):
    temp_test_xdata = pd.read_csv(path + 'test_data.csv')
    temp_test_ydata = pd.read_csv(path + 'test_label.csv')
    test_data = _construct_xdata(temp_test_xdata)
    test_label = _construct_ydata(temp_test_ydata)
    return test_data, test_label

if __name__ == '__main__':
    train_data = get_train_data_array_db(65)
    train_data = get_train_data_array_csv(65)
    data, label = construct_data_for_lstm(train_data)
    train_data_split(data, label)
