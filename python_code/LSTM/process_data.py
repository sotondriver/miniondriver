# -*- coding: utf-8 -*-
"""
Created on 16/6/6 15:35 2016

@author: harry sun
"""
import numpy as np
import pandas as pd


def _construct_xdata(data):
    """
    data should be pd.DataFrame()
    """
    data1 = data.values.tolist()
    temp = []

    for entry in data1:
        matrix = []
        entry = entry[0]
        entry = entry.split(',')
        entry =  map(int, entry)
        time = entry[0]
        matrix.append([time-3] + entry[1:67])
        matrix.append([time-2] + entry[67:133])
        matrix.append([time-1] + entry[133:199])
        temp.append(matrix)

    data_out = np.array(temp)
    return data_out


def _construct_ydata(data):
    data1 = data.values.tolist()
    temp = []
    for entry in data1:
        entry = entry[0]
        entry = entry.split(',')
        entry = map(int, entry)
        temp.append(entry)
    data_out = np.array(temp)
    return data_out

def clean_data(x, y):
    non_zero_idx = np.where(y > 0)[0]
    size = len(non_zero_idx)
    x_out = x[non_zero_idx]
    y_out = y[non_zero_idx]
    return x_out, y_out, size

def train_data_split(train_list, label_list, seed, train_size=0.7, test_size=0.1):
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

    X_train = _construct_xdata(train_data.iloc[0:idx1])
    y_train = _construct_ydata(label_data.iloc[0:idx1])
    x_validate = _construct_xdata(train_data.iloc[idx1:idx2])
    y_validate = _construct_ydata(label_data.iloc[idx1:idx2])
    X_test = train_data.iloc[idx2:]
    y_test = label_data.iloc[idx2:]

    return (X_train, y_train), (x_validate, y_validate), (X_test, y_test)


def load_test_data(path):
    temp_test_xdata = pd.read_csv(path + 'test_data.csv')
    temp_test_ydata = pd.read_csv(path + 'test_label.csv')
    test_data = _construct_xdata(temp_test_xdata)
    test_label = _construct_ydata(temp_test_ydata)
    return test_data, test_label

