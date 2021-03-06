# -*- coding: utf-8 -*-
"""
Created on 16/6/6 15:35 2016

@author: harry sun
"""
import linecache
import string

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

# from python_code.preprocess.create_training_data import get_order_data_array_db

predict_time_slot_window = [43, 44, 45, 46, 55, 56, 57, 58, 67, 68, 69, 70, 79, 80, 81, 82, 91, 92, 93, 94, 103, 104,
                            105, 106, 115, 116, 117, 118, 127, 128, 129, 130, 139, 140, 141, 142]
# idx = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 21, 22, 25, 26, 27, 29, 30, 31, 32, 33, 34, 35, 36,
#        38, 39, 40, 41, 42, 43, 44, 45, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]



#
# def get_train_data_array_csv(district_idx):
#     order_path = '../../processed_data/train/D'+str(district_idx)+'_order_data.csv'
#     traffic_path = '../../processed_data/train/D'+str(district_idx)+'_traffic_data.csv'
#     weather_path = '../../processed_data/train/weather_data'
#
#     order_data = pd.read_csv(order_path, header=None).values
#     traffic_data = pd.read_csv(traffic_path, header=None).values[..., 1:5]
#     weather_data = pd.read_csv(weather_path, header=None).values[..., 1:4]
#     train_data = np.concatenate((order_data, traffic_data), axis=1)
#
#     normalised_data = scale(train_data, axis=1, copy=True)
#     normalised_data[..., 2] = train_data[..., 2]
#     return normalised_data


def get_train_data_array_csv(idx):
    total_data = np.zeros((1, 36))
    for district_idx in idx:
        order_path = '../../processed_data/train/D' + str(district_idx) + '_order_data.csv'
        traffic_path = '../../processed_data/train/D' + str(district_idx) + '_traffic_data.csv'
        weather_path = '../../processed_data/train/weather_data.csv'
        poi_path = '../../processed_data/train/poi_data.csv'

        order_data = pd.read_csv(order_path, header=None).values
        traffic_data = pd.read_csv(traffic_path, header=None).values[..., 1:5]
        weather_data = pd.read_csv(weather_path, header=None).values[..., 1:4]
        temp_poi_data = pd.read_csv(poi_path, header=None).values[district_idx - 1:district_idx, 1:26]
        poi_data = np.tile(temp_poi_data, (3024, 1))
        train_data = np.concatenate((order_data, traffic_data, weather_data, poi_data), axis=1)

        total_data = np.concatenate((total_data, train_data), axis=0)
    total_data = total_data[1:, ...]

    normalised_data = scale(total_data, axis=1, copy=True)
    normalised_data[..., 0:4] = total_data[..., 0:4]

    dim = total_data.shape[1]
    return normalised_data, dim


def get_test_data_array_csv(idx):
    total_data = np.zeros((1, 36))
    for district_idx in idx:
        order_path = '../../processed_data/test/D' + str(district_idx) + '_order_data.csv'
        traffic_path = '../../processed_data/test/D' + str(district_idx) + '_traffic_data.csv'
        weather_path = '../../processed_data/test/weather_data.csv'
        poi_path = '../../processed_data/test/poi_data.csv'

        order_data = pd.read_csv(order_path, header=None).values
        traffic_data = pd.read_csv(traffic_path, header=None).values[..., 1:5]
        weather_data = pd.read_csv(weather_path, header=None).values[..., 1:4]
        temp_poi_data = pd.read_csv(poi_path, header=None).values[district_idx - 1:district_idx, 1:26]
        poi_data = np.tile(temp_poi_data, (135, 1))
        test_data = np.concatenate((order_data, traffic_data, weather_data, poi_data), axis=1)

        total_data = np.concatenate((total_data, test_data), axis=0)
    total_data = total_data[1:, ...]

    normalised_data = scale(total_data, axis=1, copy=True)
    normalised_data[..., 0:4] = total_data[..., 0:4]

    dim = total_data.shape[1]
    return normalised_data, dim


def get_train_data_array_csv_by_active_matrix(district_idx):
    order_path = '../../processed_data/train/D' + str(district_idx) + '_order_data.csv'
    traffic_path = '../../processed_data/train/D' + str(district_idx) + '_traffic_data.csv'
    weather_path = '../../processed_data/train/weather_data.csv'
    poi_path = '../../processed_data/train/poi_data.csv'
    train_data = pd.read_csv(order_path, header=None).values

    matrix_path = '../../processed_data/coef_active_matrix.csv'
    line = linecache.getline(matrix_path, district_idx)
    line = line.strip('\n')
    line_list = line.split(',')
    for (idx, item) in enumerate(line_list):
        if (item == '1') & (idx + 1 != district_idx):
            active_district_id = idx + 1
            active_path = '../../processed_data/train/D' + str(active_district_id) + '_order_data.csv'
            active_data = pd.read_csv(active_path, header=None).values[..., 2:3]
            train_data = np.concatenate((train_data, active_data), axis=1)

    traffic_data = pd.read_csv(traffic_path, header=None).values[..., 1:5]
    weather_data = pd.read_csv(weather_path, header=None).values[..., 1:4]
    temp_poi_data = pd.read_csv(poi_path, header=None).values[district_idx - 1:district_idx, 1:26]
    poi_data = np.tile(temp_poi_data, (3024, 1))
    train_data = np.concatenate((train_data, traffic_data, weather_data), axis=1)

    normalised_data = scale(train_data, axis=1, copy=True)

    normalised_data[..., 0:4] = train_data[..., 0:4]
    # normalised_data[..., 0] = train_data[..., 0]
    # normalised_data[..., 2] = train_data[..., 2]
    dim = train_data.shape[1]

    test_idx = get_data_by_test_window()
    normalised_data = normalised_data[test_idx, ...]
    return normalised_data, dim


def get_data_by_test_window(total_len):
    idx_list = []
    for j in range(total_len):
        for i in range(1, 21, 1):
            for time_slot_idx in predict_time_slot_window:
                idx_list.append(i * 144 + time_slot_idx - 1)
    return idx_list


def construct_data_for_lstm(data):
    data1 = data.tolist()
    _data = []
    _label = []
    for idx in range(data.shape[0] - 3):
        matrix = []
        temp_idx = idx
        matrix.append(data1[temp_idx])
        matrix.append(data1[temp_idx + 1])
        matrix.append(data1[temp_idx + 2])
        # matrix.append(data1[temp_idx + 2])
        # matrix.append(data1[temp_idx + 1])
        # matrix.append(data1[temp_idx])
        _label.append(data1[temp_idx + 3])
        _data.append(matrix)

    return _data, _label


def construct_test_data_for_lstm(data):
    data1 = data.tolist()
    _data = []
    for idx in range(0, data.shape[0], 3):
        matrix = []
        temp_idx = idx
        matrix.append(data1[temp_idx])
        matrix.append(data1[temp_idx + 1])
        matrix.append(data1[temp_idx + 2])
        _data.append(matrix)

    return _data


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
    t = []
    for item in temp_test_xdata.values:
        item1 = []
        for s in item:
            l = s.translate(None, '[]')
            l = l.split(',')
            s = np.asarray(l, dtype=np.float32)
            item1.append(s)
        t.append(item1)
    test_data = np.asarray(t)

    test_label = temp_test_ydata.values
    return test_data, test_label


if __name__ == '__main__':
    get_test_data_array_csv()
    load_test_data('../../result/attempt6/lr_0.002_loss_2_batch_ratio_0.002/')

    for district_id in range(1, 66 + 1, 1):
        # train_data = get_train_data_array_db(65)
        _, dim = get_train_data_array_csv_by_active_matrix(district_id)
        # weight = np.ones(shape=(dim)) * 0.01
        # weight[2] = weight[2]*10
        print('%d  %d' % (district_id, dim))
        # train_data = get_train_data_array_csv(1)
        # data, label = construct_data_for_lstm(train_data)
        # train_data_split(data, label)
