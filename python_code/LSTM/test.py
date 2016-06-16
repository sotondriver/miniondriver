# -*- coding: utf-8 -*-
"""
Created on 16/6/9 17:28 2016

@author: harry sun
"""
import linecache

from LSTM import initial_lstm_model, initial_lstm_model
from extend_function import listdir_no_hidden, write_list_to_csv
from process_data import load_test_data, get_test_data_array_csv, construct_data_for_lstm, get_train_data_array_csv, \
    construct_test_data_for_lstm, idx
import numpy as np
import pandas as pd


mape_sum = 0
mape_num = 0


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


def calculate_test_result(attempt_path):
    mape_list = []
    csv_path = attempt_path + 'LSTM_MAPE_list.csv'
    csv_list = open(csv_path).readlines()
    for idx in range(66):
        entry = csv_list[idx]
        entry = entry.strip('\n')
        entry_list = entry.split(',')
        # initial model and load model
        model = initial_lstm_model(entry_list[1], int(entry_list[2]))
        model_path = attempt_path + 'model_district_'+str(idx+1)+'.h5'
        model.load_weights(model_path)
        # load test data
        test_data, test_label = load_test_data(attempt_path)
        predicted = model.predict(test_data)
        # get the colomn of test label
        temp_test_label = test_label[..., 2]
        mape = return_mape(predicted, temp_test_label)
        mape_list.append([idx+1]+[entry_list[0]]+[mape])
        print(('District: %d %.8s %f') % (idx+1, entry_list[0], mape))
    mape_list.append([67]+['Overall mape']+[mape_sum/mape_num])
    write_list_to_csv(mape_list, attempt_path+'TEST_MAPE_list.csv')
    print(('Overall mape: %f') % (mape_sum/mape_num))


def generate_test_result(attempt_path):
    # load the test data
    test_array, data_dim = get_test_data_array_csv()
    test_data = construct_test_data_for_lstm(test_array)
    csv_path = attempt_path + 'LSTM_MAPE_list.csv'
    entry = open(csv_path).readlines()[0]
    entry = entry.strip('\n')
    entry_list = entry.split(',')
    # initial model and load model
    model = initial_lstm_model(entry_list[1], int(entry_list[2]))
    model_path = attempt_path + 'model_district_1.h5'
    model.load_weights(model_path)
    predicted = model.predict(test_data)
    predict = predicted.flatten().tolist()

    result_list = return_predict_label_with_idx(predict)
    write_list_to_csv(result_list, attempt_path + 'result.csv')


def return_predict_label_with_idx(predict):
    date_list = [23,25,27,29,31]
    predict_time_slot = [46,58,70,82,94,106,118,130,142]
    new_list = []
    count = 0
    for district_idx in idx:
        for date_idx in date_list:
            for time_slot_idx in predict_time_slot:
                if ((date_idx == 25) | (date_idx == 29)) & (time_slot_idx == 46):
                    pass
                else:
                    new_list.append([district_idx]+['2016-01-'+str(date_idx)+'-'+str(time_slot_idx)]+[predict[count]])
                count +=1
    return new_list


def return_predict_label_with_date(predict, district_id):
    test_path = '../../processed_data/didi_test_data.csv'
    test_list_str = open(test_path).readlines()
    new_list = []
    if len(predict.shape) > 1:
        predict_list = predict.flatten().tolist()
    else:
        predict_list = predict.tolist()
    for idx in range(len(predict_list)):
        if idx < 9:
            date = '2016-01-22-'
        elif idx < 17:
            date = '2016-01-24-'
        elif idx < 26:
            date = '2016-01-26-'
        elif idx < 34:
            date = '2016-01-28-'
        else:
            date = '2016-01-30-'
        line = test_list_str[idx]
        time_slot = line.split(',')[0]
        date1 = date + str(time_slot)
        new_list.append([district_id+1]+[date1]+[predict_list[idx]])
    return new_list


if __name__ == '__main__':
    MODEL_OUT_PATH = '../../result/attempt1/'
    #
    dir_list = listdir_no_hidden(MODEL_OUT_PATH)
    parent_path = MODEL_OUT_PATH + dir_list[0]+'/'
    #
    dir_list1 = listdir_no_hidden(parent_path)
    sub_path = parent_path + dir_list1[0]+'/'
    #
    # calculate_test_result(parent_path)
    generate_test_result(parent_path)