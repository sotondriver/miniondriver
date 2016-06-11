# -*- coding: utf-8 -*-
"""
Created on 16/6/6 21:32 2016

@author: harry sun
"""
import os
import time
import keras
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense, Flatten
from keras.layers.recurrent import LSTM
from keras.regularizers import l2, activity_l2
from extend_function import write_list_to_csv, save_test_csv
from process_data import *

# parameters for tuned
attempt = 2
seed = 5
batch_size_ratio = 0.3
initial_lr = 0.1
early_stop_patience = 30
fit_validation_split = 0.2
fit_epoch = 200
activator_list = ['linear', 'linear']

# for debug visualize
checkpointer_verbose = 1
early_stop_verbose = 1
fit_verbose = 0

# path for using
PARENT_IN_PATH = '../../processed_data/'
MODEL_OUT_PATH = '../../result/attempt'+str(attempt)+'_batch_ratio_'+str(batch_size_ratio)+'/'
train_path = PARENT_IN_PATH + 'didi_train_data.csv'
label_path = PARENT_IN_PATH + 'didi_train_label.csv'

# global variables
mape_sum = 0
mape_num = 0


def initial_lstm_moel1():
    data_dim = 67
    timesteps = 3
    layers = [3, ]

def initial_lstm_model(activator):
    data_dim = 67
    timesteps = 3

    model = Sequential()

    model.add(LSTM(128, input_shape=(timesteps, data_dim),
                   return_sequences=True, W_regularizer=l2(0.01)))
    model.add(Activation(activator))
    model.add(Dropout(0.25))
    model.add(TimeDistributedDense(input_dim=timesteps, output_dim=3))
    #

    # model.add(LSTM(256, return_sequences=True, W_regularizer=l2(0.05)))
    # model.add(Activation(activator))
    # model.add(Dropout(0.25))
    # model.add(TimeDistributedDense(input_dim=timesteps, output_dim=1))
    # #
    #
    # model.add(LSTM(128, return_sequences=True, W_regularizer=l2(0.05)))
    # model.add(Activation(activator))
    # model.add(Dropout(0.25))
    # model.add(TimeDistributedDense(input_dim=timesteps, output_dim=1))

    model.add(LSTM(64, return_sequences=True, W_regularizer=l2(0.01)))
    model.add(Activation(activator))
    model.add(Dropout(0.25))
    model.add(TimeDistributedDense(input_dim=timesteps, output_dim=1))

    model.add(Flatten())

    # model.add(Dense(64))
    # model.add(Dropout(0.25))
    # model.add(Activation(activator))
    # # #
    model.add(Dense(32))
    model.add(Dropout(0.25))
    model.add(Activation(activator))
    # # #
    model.add(Dense(16))
    model.add(Dropout(0.25))
    model.add(Activation(activator))
    #
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
    checkpointer = ModelCheckpoint(MODEL_OUT_PATH+'model_district_'+str(district_id+1)+'.h5',
                                   verbose=checkpointer_verbose, save_best_only=True)
    earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stop_patience,
                                                  verbose=early_stop_verbose, mode='min')
    model.fit(x_train, y_train, verbose=fit_verbose, validation_split=fit_validation_split,
              batch_size=batch_size, shuffle=True,
              nb_epoch=fit_epoch, callbacks=[decay_lr, earlystopping, checkpointer])


def return_mape(predict_result, true_result):
    global mape_sum, mape_num
    predict_result1 = predict_result.flatten()
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


def multi_model(x_train, y_train, x_validate, y_validate):
    mape_list = []
    for district_id in range(66):
        mape_entry = []
        model_list = []
        temp_x_train, temp_y_train, size = clean_data(x_train, y_train[..., district_id])
        temp_x_validate, temp_y_validate, _ = clean_data(x_validate, y_validate[..., district_id])
        start = time.time()
        for activator in activator_list:
            model = initial_lstm_model(activator=activator)
            train_model(model, temp_x_train, temp_y_train, district_id, size)
            model.load_weights(MODEL_OUT_PATH+'model_district_'+str(district_id+1)+'.h5')
            predicted = model.predict(temp_x_validate)
            model_list.append(model)
            mape_entry.append(return_mape(predicted, temp_y_validate))
        end = time.time()
        best_idx = mape_entry.index(min(mape_entry))
        best_model = model_list[best_idx]
        best_model.save_weights(MODEL_OUT_PATH+'model_district_'+str(district_id+1)+'.h5', overwrite=True)
        print('District: '+str(district_id+1)+' '+str(mape_entry)+' Choose: '+activator_list[best_idx])
        print(' Time: %.2f minutes' % ((end - start) / 60))
        mape_list.append([mape_entry[best_idx]] + [activator_list[best_idx]])
    mape_list.append(['overall mape']+[mape_sum / mape_num])
    return mape_list


if __name__ == '__main__':
    st_time = time.time()
    d = os.path.dirname(MODEL_OUT_PATH)
    if not os.path.exists(d):
        os.makedirs(d)
    train_data_str = open(train_path)
    label_data_str = open(label_path)
    train_list_str = train_data_str.readlines()
    label_list_str = label_data_str.readlines()
    (train_data, train_label), (validate_data, validate_label), (test_data, test_label) \
        = train_data_split(train_list_str, label_list_str, seed=seed)
    # save the test_data into the models directory
    save_test_csv(MODEL_OUT_PATH, test_data, test_label)
    mape = multi_model(train_data, train_label, validate_data, validate_label)
    out_path = MODEL_OUT_PATH + 'LSTM_MAPE_list.csv'
    # save the validation MAPE for every model and overall MAPE
    write_list_to_csv(mape, out_path)
    ed_time = time.time()
    print(' Overall Time: %.2f hours' % ((ed_time - st_time) / 3600))
    print 'overall mape loss: %f\n' % (mape_sum / mape_num)
