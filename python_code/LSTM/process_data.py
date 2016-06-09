# -*- coding: utf-8 -*-
"""
Created on 16/6/6 15:35 2016

@author: harry sun
"""
import numpy as np
import pandas as pd


def _construct_Xdata(data):
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
    x_out = x[non_zero_idx]
    y_out = y[non_zero_idx]
    return x_out, y_out

def train_test_split(train_list, label_list, seed, test_size=0.2):
    """
    This just splits data to training and testing parts
    """
    data_ = range(len(train_list))
    label_ = range(len(train_list))
    shuffle_idx = range(len(train_list))
    np.random.seed(641)
    np.random.shuffle(shuffle_idx)
    for i in range(len(shuffle_idx)):
        idx_ = shuffle_idx[i]
        data_[idx_] = train_list[idx_]
        label_[idx_] = label_list[idx_]

    ntrn = int(round(len(train_list) * (1 - test_size)))

    train_data = pd.DataFrame(data_)
    label_data = pd.DataFrame(label_)

    X_train = _construct_Xdata(train_data.iloc[0:ntrn])
    y_train = _construct_ydata(label_data.iloc[0:ntrn])
    X_test = _construct_Xdata(train_data.iloc[ntrn:])
    y_test = _construct_ydata(label_data.iloc[ntrn:])

    return (X_train, y_train), (X_test, y_test)

