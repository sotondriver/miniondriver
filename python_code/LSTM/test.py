# -*- coding: utf-8 -*-
"""
Created on 16/6/9 17:28 2016

@author: harry sun
"""
import linecache

from LSTM import initial_lstm_model
from extend_function import listdir_no_hidden, write_list_to_csv
from process_data import load_test_data
import numpy as np

MODEL_OUT_PATH = '../../result/'
result_list = listdir_no_hidden(MODEL_OUT_PATH)
sub_path = MODEL_OUT_PATH + result_list[0] + '/'
sub_path1 = MODEL_OUT_PATH + result_list[-1] + '/'
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

def calculate_test_result():
    csv_path = sub_path + 'LSTM_MAPE_list.csv'

    csv_list = open(csv_path).readlines()
    for idx in range(66):
        entry = csv_list[idx]
        entry = entry.strip('\n')
        entry_list = entry.split(',')
        model = initial_lstm_model(entry_list[1])
        model_path = sub_path + 'model_district_'+str(idx+1)+'.h5'
        model.load_weights(model_path)
        test_data, test_label = load_test_data(sub_path1)
        predicted = model.predict(test_data)
        # get the colomn of test label
        temp_test_label = test_label[..., idx]
        mape = return_mape(predicted, temp_test_label)
        print('District: '+str(idx+1)+' '+ entry_list[0] + '   %f') % mape
    print('Overall mape: %f') % (mape_sum/mape_num)


def choose_best_model():
    global result_list, sub_path1
    best_mape_list = []
    for model_idx in range(66):
        mape_list = []
        model_list = []
        activator_list = []
        test_data, test_label = load_test_data(sub_path1)
        temp_test_label = test_label[..., model_idx]
        for path_idx in range(11):
            result_sub_path = MODEL_OUT_PATH + result_list[path_idx]+'/'
            csv_path = result_sub_path + 'LSTM_MAPE_list.csv'
            line = linecache.getline(csv_path, model_idx+1)
            line = line.strip('\n')
            line_list = line.split(',')
            activator = line_list[1]
            model = initial_lstm_model(activator)
            model_path = result_sub_path + 'model_district_' + str(model_idx + 1) + '.h5'
            model.load_weights(model_path)
            predicted = model.predict(test_data)
            mape = return_mape(predicted, temp_test_label)
            # save all the result
            activator_list.append(activator)
            mape_list.append(mape)
            model_list.append(model)
        best_idx = mape_list.index(min(mape_list))
        # print result
        print('District: %d %f  '+result_list[best_idx]) %(model_idx+1, mape_list[best_idx])
        best_mape_list.append([mape_list[best_idx]] + [activator_list[best_idx]])
        best_model = model_list[best_idx]
        best_model.save_weights('../../final_result/attempt1/' + 'model_district_' + str(model_idx + 1) + '.h5', overwrite=True)
    write_list_to_csv(best_mape_list, '../../final_result/attempt1/'+'LSTM_MAPE_list.csv')

if __name__ == '__main__':
    choose_best_model()

