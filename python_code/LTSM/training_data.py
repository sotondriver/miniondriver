# -*- coding: utf-8 -*-
"""
Created on 16/6/6 21:32 2016

@author: harry sun
"""
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense, Flatten
from keras.layers.recurrent import LSTM


PARENT_OUT_PATH = '../../processed_data/'
train_path = PARENT_OUT_PATH + 'didi_train_data.csv'
label_path = PARENT_OUT_PATH + 'didi_train_label.csv'


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
        entry =  map(float, entry)
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
        entry = map(float, entry)
        temp.append(entry)
    data_out = np.array(temp)
    return data_out

def train_test_split(train_list, label_list, test_size=0.1):
    """
    This just splits data to training and testing parts
    """
    data_ = range(len(train_list))
    label_ = range(len(train_list))
    shuffle_idx = range(len(train_list))
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

def predict_by_LSTM(X_train, y_train, X_test, y_test):
    data_dim = 67
    timesteps = 3

    model = Sequential()

    # model.add(Dense(100, input_dim=3))
    # model.add(Dropout(0.5))
    # model.add(Activation("linear"))

    model.add(LSTM(128, input_shape=(timesteps, data_dim), return_sequences=True))
    model.add(Activation("linear"))
    model.add(Dropout(0.25))
    model.add(TimeDistributedDense(input_dim=timesteps, output_dim=1))
    #
    # model.add(LSTM(64, return_sequences=True))
    # model.add(Activation("linear"))
    # model.add(Dropout(0.5))
    # model.add(TimeDistributedDense(input_dim=timesteps, output_dim=1))

    # model.add(LSTM(128, return_sequences=True))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    #
    model.add(Dense(32))
    model.add(Dropout(0.5))
    model.add(Activation("linear"))
    # #
    model.add(Dense(16))
    model.add(Dropout(0.25))
    model.add(Activation("linear"))
    # #
    model.add(Dense(1))
    model.add(Activation("linear"))

    # model.compile(loss='mape', optimizer='Adam', metrics=['accuracy'])
    model.compile(loss='mape', optimizer='Adam')
    # model.fit(X_train, y_train, verbose=False, batch_size=100, nb_epoch=20, validation_split=0.1)
    model.fit(X_train, y_train, verbose=False, batch_size=100, nb_epoch=20)
    predicted = model.predict(X_test)

    predict_result = predicted[0]
    test_label = y_test
    a = np.abs(test_label - predict_result)
    sum_ = 0
    num_ = 0
    for j in range(len(predict_result)):
        if test_label[j] != 0:
            sum_ = sum_ + a[j] / test_label[j]
            num_ = num_ + 1
    if num_ == 0:
        print('No num')
    else:
        print (sum_ / num_)
    return sum_, num_

if __name__ == '__main__':
    temp_sum_ = 0
    temp_num_ = 0
    temp_b_ = 0
    train_data = open(train_path)
    label_data = open(label_path)
    train_list = train_data.readlines()
    label_list = label_data.readlines()
    (X_train, y_train), (X_test, y_test) = train_test_split(train_list, label_list)
    for i in range(66):
        temp_y_train = y_train[...,i]
        temp_y_test = y_test[...,i]
        sum_, num_= predict_by_LSTM(X_train, temp_y_train, X_test, temp_y_test)
        temp_sum_ += sum_
        temp_num_ += num_
    print 'mape loss: %f\n' % (temp_sum_ / temp_num_)