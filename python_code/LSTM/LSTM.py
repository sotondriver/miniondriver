# -*- coding: utf-8 -*-
"""
Created on 16/6/6 21:32 2016

@author: harry sun
"""
import os
import time
import keras
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import TimeDistributed, GRU
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense, Flatten ,RepeatVector
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
from keras.regularizers import l1l2, l1, activity_l2
from extend_function import write_list_to_csv, save_test_csv
from process_data import clean_zeros, get_train_data_array_csv, construct_data_for_lstm, \
    train_data_split, get_train_data_array_csv_by_active_matrix

# idx = [51]
# idx1 = [37,46]
# # idx1 = [7, 8, 14, 20, 23, 24, 28, 37, 46, 48, 51]
# idx2 = [1, 4, 6, 12, 19, 21, 22, 26, 27, 42]
# idx3 = [37, 46]
# idx4 = [2, 3, 5, 9, 10, 11, 13, 15, 16, 17, 18, 25, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 43, 44,
#         45, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]
# idx5 = [7,48]
# idx6 = [28]
# idx7 = [14,20,24]
# idx8 = [8,23]

# parameters for tuning

batch_size_ratio = 0.002
# batch_size = 1
initial_lr = 0.002
early_stop_patience = 10
fit_validation_split = 0.2
fit_epoch = 200
activator_list = ['linear']
clean_zeros_flag = True
fit_monitor = 'val_loss'

# for debug visualize
checkpointer_verbose = 0
early_stop_verbose = 0
fit_verbose = 0

# path for using
PARENT_IN_PATH = '../../processed_data/'

train_path = PARENT_IN_PATH + 'didi_train_data.csv'
label_path = PARENT_IN_PATH + 'didi_train_label.csv'

# global variables
mape_sum = 0
mape_num = 0


def initial_lstm_model(activator, data_dim):
    timesteps = 3

    model = Sequential()
    if data_dim > 30:

        model.add(LSTM(32, input_shape=(timesteps, data_dim), dropout_W=0.25, dropout_U=0.25,
                       return_sequences=True, W_regularizer=l1(0.01), U_regularizer=l1(0.01)))
        model.add(Activation(activator))
        model.add(Dropout(0.25))
        # model.add(TimeDistributed(Dense(output_dim=64, W_regularizer=l1(0.1),b_regularizer=l1(0.1))))
        #

        model.add(LSTM(16, return_sequences=True, dropout_W=0.25, dropout_U=0.25,
                       W_regularizer=l1(0.01), U_regularizer=l1(0.01)))
        model.add(Activation(activator))
        model.add(Dropout(0.25))
        # model.add(TimeDistributed(Dense(output_dim=16, activation=activator, W_regularizer=l1(0.01))))

    else:
        model.add(LSTM(32, input_shape=(timesteps, data_dim), dropout_W=0.25, dropout_U=0.25,
                       return_sequences=True, W_regularizer=l1(0.1), U_regularizer=l1(0.1)))
        model.add(Activation(activator))
        model.add(Dropout(0.25))
        # model.add(TimeDistributed(Dense(output_dim=64, activation=activator, W_regularizer=l1l2(0.1, 0.01))))
        #

    model.add(LSTM(16, return_sequences=False, dropout_W=0.25, dropout_U=0.25,
                   W_regularizer=l1(0.01), U_regularizer=l1(0.01)))
    model.add(Activation(activator))
    model.add(Dropout(0.25))

    model.add(Dense(8))
    model.add(Dropout(0.25))
    model.add(Activation(activator))
    # #
    model.add(Dense(1))
    model.add(Activation(activator))

    optimizer = keras.optimizers.Adam(lr=initial_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mape', optimizer=optimizer)

    return model


def train_model(model, x_train, y_train, district_id, size):
    class ProcessControl(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []

        def on_epoch_begin(self, epoch, logs={}):
            new_lr = np.float32(model.optimizer.lr.get_value() * 0.97)
            model.optimizer.lr.set_value(new_lr)
            # print new_lr

    batch_size = int(np.math.ceil(size * batch_size_ratio))
    decay_lr = ProcessControl()
    checkpointer = ModelCheckpoint(MODEL_OUT_PATH+'model_district_'+str(district_id)+'.h5',
                                   verbose=checkpointer_verbose, save_best_only=True)
    earlystopping = keras.callbacks.EarlyStopping(monitor=fit_monitor, patience=early_stop_patience,
                                                  verbose=early_stop_verbose, mode='min')
    model.fit(x_train, y_train, verbose=fit_verbose, validation_split=fit_validation_split,
              batch_size=batch_size, shuffle=True,
              nb_epoch=fit_epoch, callbacks=[decay_lr, earlystopping, checkpointer])


def return_mape(predict_result, true_result):
    global mape_sum, mape_num
    if len(predict_result.shape)>1:
        predict_result1 = predict_result.flatten()
    else:
        predict_result1 = predict_result
    true_result1 = true_result
    a = np.abs(true_result1 - predict_result1)
    _sum = 0
    _num = 0
    for j in range(len(predict_result1)):
        if true_result1[j] != 0:
            _sum = _sum + a[j] / true_result1[j]
            _num += 1
    mape_sum += _sum
    mape_num += _num
    return _sum/_num


def multi_model(x_train, y_train, x_validate, y_validate, district_id, dim):
    mape_entry = []
    model_list = []
    # clean zero gaps entry
    temp_x_train, temp_y_train, size = clean_zeros(x_train, y_train[..., 2], clean_zeros_flag)
    temp_x_validate, temp_y_validate, _ = clean_zeros(x_validate, y_validate[..., 2], clean_zeros_flag)
    #

    start = time.time()
    for activator in activator_list:
        model = initial_lstm_model(activator=activator, data_dim=dim)
        train_model(model, temp_x_train, temp_y_train, district_id, size)
        model.load_weights(MODEL_OUT_PATH+'model_district_'+str(district_id)+'.h5')
        predicted = model.predict(temp_x_validate)
        model_list.append(model)
        mape_entry.append(return_mape(predicted, temp_y_validate))
    end = time.time()
    best_idx = mape_entry.index(min(mape_entry))
    best_model = model_list[best_idx]
    best_model.save_weights(MODEL_OUT_PATH+'model_district_'+str(district_id)+'.h5', overwrite=True)
    print('District: '+str(district_id)+' '+str(mape_entry)+' Choose: '+activator_list[best_idx])
    print(' Time: %.2f minutes' % ((end - start) / 60))
    return [mape_entry[best_idx]] + [activator_list[best_idx]] + [dim]


if __name__ == '__main__':
    path = '../../processed_data/cluster_8.csv'
    cluster = pd.read_table(path, header=None).values
    for i in range(6,7+1,1):
        cluster_idx = cluster[i].tolist()[0].split(',')
        cluster_idx = map(int, cluster_idx)
        for j in range(1, 3+1,1):

            attempt = [3, 'final_try_on_idx'+str(i)+'_'+str(j)]
            MODEL_OUT_PATH = '../../result/attempt' + str(attempt[0]) + '/' \
                             + str(attempt[1]) + '_batch_ratio_' + str(batch_size_ratio) + '/'
            mape_list = []
            st_time = time.time()
            d = os.path.dirname(MODEL_OUT_PATH)
            if not os.path.exists(d):
                os.makedirs(d)
            # get data from csv or mongodb
            data_array, train_dim = get_train_data_array_csv(cluster_idx)
            # construct data
            train_array, label_array = construct_data_for_lstm(data_array)
            # split data by 7:2:1
            (train_data, train_label), (validate_data, validate_label), (test_data, test_label) \
                = train_data_split(train_array, label_array)
            # save the test_data into the models directory
            save_test_csv(MODEL_OUT_PATH, test_data, test_label)
            mape_entry = multi_model(train_data, train_label, validate_data, validate_label, 1, train_dim)
            mape_list.append(mape_entry)
            # save the validation MAPE for every model and overall MAPE
            out_path = MODEL_OUT_PATH + 'LSTM_MAPE_list.csv'
            write_list_to_csv(mape_list, out_path)
            ed_time = time.time()
            print(' Overall Time: %.2f hours' % ((ed_time - st_time) / 3600))
            print('overall mape loss: %f\n' % (mape_sum / mape_num))
            mape_list.append(['overall mape']+[mape_sum / mape_num])
