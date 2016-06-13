# -*- coding: utf-8 -*-
"""
Created on 16/6/6 21:32 2016

@author: harry sun
"""
import os
import time
import keras
import numpy as np
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import TimeDistributed
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense, Flatten
from keras.layers.recurrent import LSTM
from keras.regularizers import l1l2, l2, activity_l2
from extend_function import write_list_to_csv, save_test_csv
from python_code.LSTM.process_data import clean_zeros_or_not, get_train_data_array_csv, construct_data_for_lstm, \
    train_data_split


# parameters for tuning
attempt = [2, 1]
seed = 5
batch_size_ratio = 1
initial_lr = 0.03/batch_size_ratio
early_stop_patience = 20
fit_validation_split = 0.2
fit_epoch = 200
activator_list = ['linear','tanh']
clean_zeros_flag = True

# for debug visualize
checkpointer_verbose = 0
early_stop_verbose = 0
fit_verbose = 0

# path for using
PARENT_IN_PATH = '../../processed_data/'
MODEL_OUT_PATH = '../../result/attempt'+str(attempt[0])+'/'\
                 +str(attempt[1])+'_batch_ratio_'+str(batch_size_ratio)+'/'
train_path = PARENT_IN_PATH + 'didi_train_data.csv'
label_path = PARENT_IN_PATH + 'didi_train_label.csv'

# global variables
mape_sum = 0
mape_num = 0


def initial_lstm_model(activator):
    data_dim = 4
    timesteps = 3

    model = Sequential()

    model.add(LSTM(8, input_shape=(timesteps, data_dim), dropout_W=0.25,
                   return_sequences=True, W_regularizer=l1l2(0.1, 0.01)))
    model.add(Activation(activator))
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Dense(output_dim=8, activation=activator, W_regularizer=l1l2(0.1, 0.01))))
    #

    model.add(LSTM(4, return_sequences=True, W_regularizer=l1l2(0.1, 0.01), dropout_W=0.25))
    model.add(Activation(activator))
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Dense(output_dim=4, activation=activator, W_regularizer=l1l2(0.1, 0.01))))

    model.add(LSTM(1, return_sequences=False, W_regularizer=l1l2(0.1, 0.01), dropout_W=0.25))
    model.add(Activation(activator))
    model.add(Dropout(0.25))
    # model.add(Flatten())
    # # #
    # model.add(Dense(32))
    # model.add(Dropout(0.25))
    # model.add(Activation(activator))
    # # # #
    # model.add(Dense(8))
    # model.add(Dropout(0.25))
    # model.add(Activation(activator))
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

    batch_size = int(round(size * batch_size_ratio))
    decay_lr = ProcessControl()
    checkpointer = ModelCheckpoint(MODEL_OUT_PATH+'model_district_'+str(district_id)+'.h5',
                                   verbose=checkpointer_verbose, save_best_only=True)
    earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stop_patience,
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


def multi_model(x_train, y_train, x_validate, y_validate, district_id):
    mape_entry = []
    model_list = []
    # clean zero gaps entry
    temp_x_train, temp_y_train, size = clean_zeros_or_not(x_train, y_train[..., 2], clean_zeros_flag)
    temp_x_validate, temp_y_validate, _ = clean_zeros_or_not(x_validate, y_validate[..., 2], clean_zeros_flag)
    #

    start = time.time()
    for activator in activator_list:
        model = initial_lstm_model(activator=activator)
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
    return [mape_entry[best_idx]] + [activator_list[best_idx]]


if __name__ == '__main__':
    mape_list = []
    st_time = time.time()
    d = os.path.dirname(MODEL_OUT_PATH)
    if not os.path.exists(d):
        os.makedirs(d)
    for district_idx in range(1, 66+1, 1):
        # get data from csv or mongodb
        data_array = get_train_data_array_csv(district_idx)
        # construct data
        train_array, label_array = construct_data_for_lstm(data_array)
        # split data by 7:2:1
        (train_data, train_label), (validate_data, validate_label), (test_data, test_label) \
            = train_data_split(train_array, label_array)
        # save the test_data into the models directory
        save_test_csv(MODEL_OUT_PATH, test_data, test_label)
        mape_entry = multi_model(train_data, train_label, validate_data, validate_label, district_idx)
        mape_list.append(mape_entry)
    # save the validation MAPE for every model and overall MAPE
    out_path = MODEL_OUT_PATH + 'LSTM_MAPE_list.csv'
    write_list_to_csv(mape_list, out_path)
    ed_time = time.time()
    print(' Overall Time: %.2f hours' % ((ed_time - st_time) / 3600))
    print('overall mape loss: %f\n' % (mape_sum / mape_num))
    mape_list.append(['overall mape']+[mape_sum / mape_num])
