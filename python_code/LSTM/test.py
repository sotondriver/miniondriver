# -*- coding: utf-8 -*-
"""
Created on 16/6/9 17:28 2016

@author: harry sun
"""
from LSTM import initial_lstm_model
from extend_function import listdir_no_hidden
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

if __name__ == '__main__':
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
