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
from keras.optimizers import SGD
import theano.tensor as T

layers = [66, 128, 300, 128, 66]
data_dim = 66
timesteps = 10

model = Sequential()

model.add(LSTM(128, input_shape=(timesteps, data_dim), return_sequences=True))
model.add(Dropout(0.25))
#
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.5))

# model.add(LSTM(128, return_sequences=True))
# model.add(Dropout(0.25))

model.add(TimeDistributedDense(input_dim=timesteps, output_dim=1))

model.add(Flatten())

model.add(Dense(66))
model.add(Dropout(0.25))

model.add(Activation("linear"))
sgd = SGD(lr=0.2, decay=1e-6, momentum=0.9, nesterov=True)


PARENT_OUT_PATH = '../../processed_data/'
path = PARENT_OUT_PATH + 'didi_train_data.csv'

def _load_data(path):
    data = []
    f = open(path)
    lines = f.readlines()
    for date in range(21):
        for idx in range(143):
            idx1 = date*141+idx
            if idx1 > 2960:
                continue
            temp_line = lines[idx1]
            line_list = temp_line.split(',')
            if idx <= 140:
                temp_gap = map(int, line_list[1:67])
                data.append(temp_gap)
            elif idx == 141:
                temp_gap = map(int, line_list[67:133])
                data.append(temp_gap)
            elif idx == 142:
                temp_gap = map(int, line_list[133:199])
                data.append(temp_gap)
    temp_data = pd.DataFrame(data)
    return temp_data


def _construct_data(data, n_prev = 10):
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def train_test_split(df, test_size=0.1):
    """
    This just splits data to training and testing parts
    """
    ntrn = int(round(len(df) * (1 - test_size)))

    X_train, y_train = _construct_data(df.iloc[0:ntrn])
    X_test, y_test = _construct_data(df.iloc[ntrn:])

    return (X_train, y_train), (X_test, y_test)


def custom_objective(y_true, y_pred):
    '''Just another crossentropy'''
    predict_result = y_pred
    test_label = y_true
    a = np.abs(test_label - predict_result)
    b = test_label - predict_result
    sum_ = 0
    num_ = 0
    for i in range(predict_result.shape[0]):
        for j in range(predict_result.shape[1]):
            if (test_label[i, j] != 0):
                sum_ = sum_ + a[i, j] / test_label[i, j]
                num_ = num_ + 1
    return sum_ / num_


if __name__ == '__main__':
    data = _load_data(path)
    model.compile(loss='mean_absolute_percentage_error', optimizer='Adam', metrics=['accuracy'])
    (X_train, y_train), (X_test, y_test) = train_test_split(data)
    t = X_train.shape[1:]
    model.fit(X_train, y_train, batch_size=100, nb_epoch=100, validation_split=0.1)
    predicted = model.predict(X_test)
    predict_result = predicted
    test_label = y_test
    a = np.abs(test_label - predict_result)
    b = test_label - predict_result
    sum_ = 0
    num_ = 0
    for i in range(predict_result.shape[0]):
        for j in range(predict_result.shape[1]):
            if (test_label[i, j] != 0):
                sum_ = sum_ + a[i, j] / test_label[i, j]
                num_ = num_ + 1
    print 'eva: %f, ls: %f\n' % (sum_ / num_, np.sqrt(np.sum(b * b)) / (predict_result.shape[0] * predict_result.shape[1]))