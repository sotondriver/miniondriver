# -*- coding: utf-8 -*-
"""
Created on 16/6/6 21:32 2016

@author: harry sun
"""
import timeit

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense, Flatten
from keras.layers.recurrent import LSTM
from keras.regularizers import l2, activity_l2
from extend_function import write_list_to_csv
from process_data import *

PARENT_OUT_PATH = '../../processed_data/'
MODEL_OUT_PATH = '../../result/attempt3/'
train_path = PARENT_OUT_PATH + 'didi_train_data.csv'
label_path = PARENT_OUT_PATH + 'didi_train_label.csv'
mape_sum = 0
mape_num = 0


def initial_lstm_model(activator):
    data_dim = 67
    timesteps = 3

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
    model.compile(loss='mape', optimizer=optimizer)

    return model

def train_model(model, x_train, y_train):
    model.fit(x_train, y_train, verbose=False, batch_size=32, nb_epoch=40)
    # model.fit(X_train, y_train, batch_size=32, nb_epoch=40)
    return model


def return_mape(predict_result, true_result):
    global mape_sum, mape_num
    predict_result1 = predict_result[0]
    a = np.abs(true_result - predict_result1)
    _sum = 0
    _num = 0
    for j in range(len(predict_result1)):
        if true_result[j] != 0:
            _sum = _sum + a[j] / true_result[j]
            _num += 1
    mape_sum += _sum
    mape_num += _num
    return _sum/_num


def multi_model(x_train, y_train, x_test, y_test):
    activator_list = ['linear', 'sigmoid', 'tanh']
    mape_list = []
    for district_id in range(66):
        mape_entry = []
        model_list = []
        temp_x_train, temp_y_train = clean_data(x_train, y_train[..., district_id])
        temp_x_test, temp_y_test = clean_data(x_test, y_test[..., district_id])
        start = timeit.timeit()
        for activator in activator_list:
            model = initial_lstm_model(activator=activator)
            model = train_model(model, temp_x_train, temp_y_train)
            predicted = model.predict(temp_x_test)
            model_list.append(model)
            mape_entry.append(return_mape(predicted, temp_y_test))
        end = timeit.timeit()
        best_idx = mape_entry.index(min(mape_entry))
        best_model = model_list[best_idx]
        best_model.save_weights(MODEL_OUT_PATH+'model_district_'+str(district_id)+'.h5', overwrite=True)
        print('District: '+str(district_id)+' '+str(mape_entry)+' Choose: '+activator_list[best_idx])
        print(' Time: '+str(end-start))
        mape_list.append([mape_entry[best_idx]] + [activator_list[best_idx]])
    return mape_list


if __name__ == '__main__':
    train_data_str = open(train_path)
    label_data_str = open(label_path)
    train_list_str = train_data_str.readlines()
    label_list_str = label_data_str.readlines()
    (train_data, train_label), (test_data, test_label) = train_test_split(train_list_str, label_list_str)
    mape = multi_model(train_data, train_label, test_data, test_label)
    out_path = PARENT_OUT_PATH + 'LSTM_MAPE_list.csv'
    write_list_to_csv(mape, out_path)
    print 'overall mape loss: %f\n' % (mape_sum / mape_num)
