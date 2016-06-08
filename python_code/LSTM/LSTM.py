# -*- coding: utf-8 -*-
"""
Created on 16/6/6 21:32 2016

@author: harry sun
"""
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense, Flatten
from keras.layers.recurrent import LSTM
from keras.regularizers import l2, activity_l2
from extend_function import write_list_to_csv
from process_data import *

PARENT_OUT_PATH = '../../processed_data/'
train_path = PARENT_OUT_PATH + 'didi_train_data.csv'
label_path = PARENT_OUT_PATH + 'didi_train_label.csv'

def train_LSTM_model(X_train, y_train, activator):
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
    # model.compile(loss='mape', optimizer='Adam', metrics=['accuracy'])
    model.compile(loss='mape', optimizer=optimizer)

    # model.fit(X_train, y_train, batch_size=32, nb_epoch=40)
    model.fit(X_train, y_train, verbose=False, batch_size=32, nb_epoch=40)

    return model

def return_mape(predict_result, true_result):
    predict_result = predict_result[0]
    test_label = true_result
    a = np.abs(test_label - predict_result)
    sum_ = 0
    num_ = 0
    for j in range(len(predict_result)):
        if test_label[j] != 0:
            sum_ = sum_ + a[j] / test_label[j]
            num_ = num_ + 1
    return sum_/num_

def multi_model(X_train, y_train, X_test, y_test):
    activator_list = ['linear', 'sigmoid', 'tanh', 'hard_sigmoid']
    mape_list = []
    for id in range(66):
        mape_entry = [id+1]
        temp_X_train, temp_y_train = clean_data(X_train, y_train[..., id])
        temp_X_test, temp_y_test = clean_data(X_test, y_test[..., id])
        for activator in activator_list:
            model = train_LSTM_model(temp_X_train, temp_y_train, activator=activator)
            predicted = model.predict(temp_X_test)
            mape_entry.append(return_mape(predicted, temp_y_test))
        print('District: '+str(mape_entry))
        mape_list.append(mape_entry)
    return mape_list


if __name__ == '__main__':
    temp_sum_ = 0
    temp_num_ = 0
    train_data = open(train_path)
    label_data = open(label_path)
    train_list = train_data.readlines()
    label_list = label_data.readlines()
    (X_train, y_train), (X_test, y_test) = train_test_split(train_list, label_list)
    mape_list = multi_model(X_train, y_train, X_test, y_test)
    write_list_to_csv(mape_list, 'LSTM_mape_list.csv', header=['district_id','linear', 'sigmoid', 'tanh', 'hard_sigmoid'])
    # print 'mape loss: %f\n' % (temp_sum_ / temp_num_)