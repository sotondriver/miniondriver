# -*- coding: utf-8 -*-
"""
Created on 16/6/8 14:52 2016

@author: harry sun
"""
import os
import pandas as pd

def write_list_to_csv(list1, path_out, header=False):
    temp = pd.DataFrame(list1)
    temp.to_csv(path_out, index=False, header=header)


def save_test_csv(path, test_data, test_label):
    test_data_out_path = path + 'test_data.csv'
    test_label_out_path = path + 'test_label.csv'

    pd.DataFrame(test_data).to_csv(test_data_out_path, index=False, header=False)
    pd.DataFrame(test_label).to_csv(test_label_out_path, index=False, header=False)


def listdir_no_hidden(path):
    list1 = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            list1.append(f)
    return list1

