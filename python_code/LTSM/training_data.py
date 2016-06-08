# -*- coding: utf-8 -*-
"""
Created on 16/6/6 21:32 2016

@author: harry sun
"""
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense, Flatten
from keras.layers.recurrent import LSTM
from keras.regularizers import l2, activity_l2
import theano.tensor as T
from keras import backend as K
from extend_function import write_list_to_csv


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

def train_test_split(train_list, label_list, test_size=0.2):
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

def mape(y_true, y_pred):
    t = K.eval(y_true.shape[0])
    # t = K.shape(y_true[y_true != 0])
    idx = K.batch_get_value(K.argmin(y_true, axis=-1))
    temp_y_true = y_true[y_true != 0]
    temp_y_pred = y_pred[y_pred > 1.0e-9]
    # diff = T.abs_((y_true - y_pred) / T.clip(T.abs_(y_true), 1.0e-9, np.inf))
    diff = T.abs_((temp_y_true - temp_y_pred) / T.abs_(temp_y_true))
    temp = T.mean(diff, axis=-1)
    return temp


def predict_by_LSTM(X_train, y_train, X_test, y_test, id):
    data_dim = 67
    timesteps = 3
    activator = 'linear'

    model = Sequential()

    # model.add(Dense(100, input_dim=3))
    # model.add(Dropout(0.5))
    # model.add(Activation("linear"))

    model.add(LSTM(128, input_shape=(timesteps, data_dim), return_sequences=True, W_regularizer=l2(0.05)))
    model.add(Activation(activator))
    model.add(Dropout(0.25))
    model.add(TimeDistributedDense(input_dim=timesteps, output_dim=1))
    #
    model.add(LSTM(64, return_sequences=True, W_regularizer=l2(0.05)))
    model.add(Activation(activator))
    model.add(Dropout(0.25))
    model.add(TimeDistributedDense(input_dim=timesteps, output_dim=1))
    #
    # model.add(LSTM(128, return_sequences=True))
    # model.add(Activation(activator))
    # model.add(Dropout(0.25))
    # model.add(TimeDistributedDense(input_dim=timesteps, output_dim=1))

    # model.add(LSTM(32, return_sequences=True))
    # model.add(Activation(activator))
    # model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(64, W_regularizer=l2(0.1), activity_regularizer=activity_l2(0.1)))
    model.add(Dropout(0.25))
    model.add(Activation(activator))
    # #
    model.add(Dense(32, W_regularizer=l2(0.1), activity_regularizer=activity_l2(0.1)))
    model.add(Dropout(0.25))
    model.add(Activation(activator))
    # # #
    model.add(Dense(16, W_regularizer=l2(0.1), activity_regularizer=activity_l2(0.1)))
    model.add(Dropout(0.25))
    model.add(Activation(activator))

    model.add(Dense(1, W_regularizer=l2(0.1), activity_regularizer=activity_l2(0.1)))
    model.add(Activation(activator))

    optimizer = keras.optimizers.Adam(lr=0.2, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # model.compile(loss='mape', optimizer='Adam', metrics=['accuracy'])
    model.compile(loss='mape', optimizer=optimizer)

    # model.fit(X_train, y_train, batch_size=32, nb_epoch=40)
    model.fit(X_train, y_train, verbose=False, batch_size=32, nb_epoch=40)
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
    print('District:'+str(id+1)+' '+str(sum_ / num_))
    return sum_, num_

if __name__ == '__main__':
    mape_list = []
    temp_sum_ = 0
    temp_num_ = 0
    train_data = open(train_path)
    label_data = open(label_path)
    train_list = train_data.readlines()
    label_list = label_data.readlines()
    (X_train, y_train), (X_test, y_test) = train_test_split(train_list, label_list)
    for i in range(66):
        temp_X_train, temp_y_train = clean_data(X_train, y_train[..., i])
        temp_X_test, temp_y_test = clean_data(X_test, y_test[..., i])
        sum_, num_= predict_by_LSTM(temp_X_train, temp_y_train, temp_X_test, temp_y_test, i)
        mape_list.append(sum_/num_)
        temp_sum_ += sum_
        temp_num_ += num_
    write_list_to_csv(mape_list, 'mape_list.csv')
    # predict_by_LSTM(X_train, y_train, X_test, y_test, 1)
    print 'mape loss: %f\n' % (temp_sum_ / temp_num_)