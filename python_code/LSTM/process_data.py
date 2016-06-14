# -*- coding: utf-8 -*-
"""
Created on 16/6/6 15:35 2016

@author: harry sun
"""
import linecache

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from python_code.preprocess.create_training_data import get_order_data_array_db


def get_train_data_array_csv(district_idx):
    order_path = '../../processed_data/train/D'+str(district_idx)+'_order_data.csv'
    traffic_path = '../../processed_data/train/D'+str(district_idx)+'_traffic_data.csv'
    weather_path = '../../processed_data/train/weather_data'

    order_data = pd.read_csv(order_path, header=None).values
    traffic_data = pd.read_csv(traffic_path, header=None).values[..., 1:5]
    weather_data = pd.read_csv(weather_path, header=None).values[..., 1:4]
    train_data = np.concatenate((order_data, traffic_data), axis=1)

    normalised_data = scale(train_data, axis=1, copy=True)
    normalised_data[..., 2] = train_data[..., 2]
    return normalised_data


def get_train_data_array_csv_by_active_matrix(district_idx):
    order_path = '../../processed_data/train/D' + str(district_idx) + '_order_data.csv'
    traffic_path = '../../processed_data/train/D' + str(district_idx) + '_traffic_data.csv'
    weather_path = '../../processed_data/train/weather_data.csv'
    poi_path = '../../processed_data/train/poi_data.csv'
    train_data = pd.read_csv(order_path, header=None).values

    matrix_path = '../../processed_data/coef_activate_matrix.csv'
    line  = linecache.getline(matrix_path, district_idx)
    line = line.strip('\n')
    line_list = line.split(',')
    for (idx, item) in enumerate(line_list):
        if (item == '1') & (idx+1 != district_idx):
            active_district_id = idx + 1
            active_path = '../../processed_data/train/D' + str(active_district_id) + '_order_data.csv'
            active_data = pd.read_csv(active_path, header=None).values[..., 2:3]
            train_data = np.concatenate((train_data, active_data), axis=1)

    traffic_data = pd.read_csv(traffic_path, header=None).values[..., 1:5]
    weather_data = pd.read_csv(weather_path, header=None).values[..., 1:4]
    temp_poi_data = pd.read_csv(poi_path, header=None).values[district_idx-1:district_idx, 1:26]
    poi_data = np.tile(temp_poi_data, (3024, 1))
    train_data = np.concatenate((train_data, traffic_data, weather_data), axis=1)

    normalised_data = scale(train_data, axis=1, copy=True)

    normalised_data[..., 0:4] = train_data[..., 0:4]
    # normalised_data[..., 0] = train_data[..., 0]
    dim = train_data.shape[1]
    return normalised_data, dim


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
        # matrix.append(data1[temp_idx+2])
        # matrix.append(data1[temp_idx+1])
        # matrix.append(data1[temp_idx])
        _label.append(data1[temp_idx+3])
        _data.append(matrix)

    return _data, _label

def clean_zeros(x, y, flag):
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
    for district_id in range(1, 66+1, 1):
        # train_data = get_train_data_array_db(65)
        _, dim = get_train_data_array_csv_by_active_matrix(district_id)
        # weight = np.ones(shape=(dim)) * 0.01
        # weight[2] = weight[2]*10
        print('%d  %d' %(district_id, dim))
        # train_data = get_train_data_array_csv(1)
        # data, label = construct_data_for_lstm(train_data)
        # train_data_split(data, label)
