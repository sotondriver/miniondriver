# -*- coding: utf-8 -*-
"""
Created on 16/6/3 00:22 2016

@author: harry sun
"""
import os
import pandas as pd


def listdir_no_hidden(path):
    list1 = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            list1.append(f)
    return list1


def write_list_to_csv(list1, path_out, header=False):
    temp = pd.DataFrame(list1)
    temp.to_csv(path_out, index=False, header=header)


def load_cluster_map():
    path = '../../training_data/cluster_map'
    temp_path = listdir_no_hidden(path)
    temp_path = path + '/' + temp_path[0]
    table = pd.read_table(temp_path, names=['district hash', 'district_id'])
    array = table.get_values()
    district_dict = {array[i][0]: array[i][1] for i in range(0, len(array), 1)}
    return district_dict


def get_time_slot(time):
    time = time.replace(' ', ':')
    time = time.replace('-', ':')
    time_list = time.split(':')
    temp1 = int(time_list[2])
    t1 = int(time_list[3])
    t2 = int(time_list[4])
    temp2 = (t1 * 6 + t2 / 10) + 1
    time_slot = [temp1, temp2]
    return time_slot


if __name__ == '__main__':
    print(get_time_slot(time='2016-01-01 23:30:22'))
    pass
