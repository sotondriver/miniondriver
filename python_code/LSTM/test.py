# -*- coding: utf-8 -*-
"""
Created on 16/6/9 17:28 2016

@author: harry sun
"""
import linecache

from LSTM import initial_lstm_model, initial_lstm_model
from extend_function import listdir_no_hidden, write_list_to_csv
from process_data import load_test_data, get_test_data_array_csv, construct_data_for_lstm, get_train_data_array_csv, \
    construct_test_data_for_lstm
import numpy as np
import pandas as pd
from python_code.SVR.modify import modify, modify1

mape_sum = 0
mape_num = 0


def return_mape(predict_result, true_result):
    global mape_sum, mape_num
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


def calculate_test_result(attempt_path, idx):
    mape_list = []
    csv_path = attempt_path + 'LSTM_MAPE_list.csv'
    csv_list = open(csv_path).readlines()
    entry = csv_list[0]
    entry = entry.strip('\n')
    entry_list = entry.split(',')
    # initial model and load model
    model = initial_lstm_model(entry_list[1], int(entry_list[2]))
    model_path = attempt_path + 'model_district_'+str(idx)+'.h5'
    model.load_weights(model_path)
    # load test data
    test_data, test_label = load_test_data(attempt_path)
    predicted = model.predict(test_data).flatten()
    # get the colomn of test label
    temp_test_label = test_label[..., 2]

    # m_predict = modify(predicted, temp_test_label)
    # #
    # predicted1 = predicted
    # for k in range(len(predicted1)):
    #     predicted1[k] = max(predicted[k] + m_predict, 1)

    mape = return_mape(predicted, temp_test_label)
    mape_list.append([idx]+[entry_list[0]]+[mape])
    print(('District: %d %.8s %f') % (idx, entry_list[0], mape))
    # mape_list.append([67]+['Overall mape']+[mape_sum/mape_num])
    # write_list_to_csv(mape_list, attempt_path+'TEST_MAPE_list.csv')
    # print(('Overall mape: %f') % (mape_sum/mape_num))


def generate_test_result(attempt_path):
    total_result_list = []
    path = '../../processed_data/cluster_8.csv'
    cluster = pd.read_table(path, header=None).values
    for i in range(8):
        cluster_idx = cluster[i].tolist()[0].split(',')
        cluster_idx = map(int, cluster_idx)
        path = attempt_path+'final_try_on_idx'+str(i)
        temp_result = np.zeros(len(cluster_idx)*45)
        for j in range(1,3+1,1):
            path1 = path + '_' + str(j) + '_batch_ratio_0.002/'
            # load the test data
            test_array, data_dim = get_test_data_array_csv(cluster_idx)
            test_data = construct_test_data_for_lstm(test_array)
            csv_path = path1 + 'LSTM_MAPE_list.csv'
            entry = open(csv_path).readlines()[0]
            entry = entry.strip('\n')
            entry_list = entry.split(',')
            # initial model and load model
            model = initial_lstm_model(entry_list[1], int(entry_list[2]))
            model_path = path1 + 'model_district_1.h5'
            model.load_weights(model_path)
            predicted = model.predict(test_data)
            predict = np.asarray(predicted.flatten().tolist())
            temp_result = np.add(temp_result,predict)
        mean_result = temp_result/3.0
        result_list = return_predict_label_with_idx(mean_result, cluster_idx)
        total_result_list += result_list
    write_list_to_csv(total_result_list, attempt_path + 'result.csv')


def return_predict_label_with_idx(predict, idx):
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


if __name__ == '__main__':
    MODEL_OUT_PATH = '../../result/attempt3/'
    #
    dir_list = listdir_no_hidden(MODEL_OUT_PATH)
    parent_path = MODEL_OUT_PATH + dir_list[1]+'/'
    #
    dir_list1 = listdir_no_hidden(parent_path)
    sub_path = parent_path + dir_list1[0]+'/'
    #
    # calculate_test_result(parent_path, 1)
    generate_test_result(MODEL_OUT_PATH)