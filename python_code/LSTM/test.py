# -*- coding: utf-8 -*-
"""
Created on 16/6/9 17:28 2016

@author: harry sun
"""
import linecache

from LSTM import initial_lstm_model, initial_lstm_model1
from extend_function import listdir_no_hidden, write_list_to_csv
from process_data import load_test_data, _construct_xdata
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
        model = initial_lstm_model(entry_list[1])
        model_path = attempt_path + 'model_district_'+str(idx+1)+'.h5'
        model.load_weights(model_path)
        # load test data
        test_data, test_label = load_test_data(attempt_path)
        predicted = model.predict(test_data)
        # get the colomn of test label
        temp_test_label = test_label[..., idx]
        mape = return_mape(predicted, temp_test_label)
        mape_list.append([idx+1]+[entry_list[0]]+[mape])
        print('District: %d %.8s %f') % (idx+1, entry_list[0], mape)
    mape_list.append([67]+['Overall mape']+[mape_sum/mape_num])
    write_list_to_csv(mape_list, attempt_path+'TEST_MAPE_list.csv')
    print('Overall mape: %f') % (mape_sum/mape_num)


def average_model_result(path):
    dir_list = listdir_no_hidden(path)
    result_list = []
    test_path = '../../processed_data/didi_test_data.csv'
    test_list_str = open(test_path).readlines()
    test_data_str = pd.DataFrame(test_list_str)
    test_data = _construct_xdata(test_data_str)
    model = initial_lstm_model('linear')
    for model_idx in range(66):
        predicted_list = []
        for path_idx in range(len(dir_list)):
            result_sub_path = path + dir_list[path_idx] + '/'
            # csv_path = result_sub_path + 'LSTM_MAPE_list.csv'
            # line = linecache.getline(csv_path, model_idx + 1)
            # line = line.strip('\n')
            # line_list = line.split(',')
            # activator = line_list[1]
            # model = initial_lstm_model(activator)
            model_path = result_sub_path + 'model_district_' + str(model_idx + 1) + '.h5'
            model.load_weights(model_path)
            # load the test data
            predicted = model.predict(test_data)
            predict = predicted.flatten().tolist()
            predicted_list.append(predict)
        predicted_mean = np.mean(predicted_list, axis=0)
        result = return_predict_label_with_date(predicted_mean, model_idx)
        # save all the result
        result_list += result
        # print result
        print('Processed District: %d') % (model_idx+1)
    write_list_to_csv(result_list, path + 'result.csv')


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
    MODEL_OUT_PATH = '../../result/'
    #
    dir_list = listdir_no_hidden(MODEL_OUT_PATH)
    parent_path = MODEL_OUT_PATH + dir_list[0]+'/'
    #
    dir_list1 = listdir_no_hidden(parent_path)
    sub_path = parent_path + dir_list1[0]+'/'
    #
    # calculate_test_result(path)
    average_model_result(parent_path)