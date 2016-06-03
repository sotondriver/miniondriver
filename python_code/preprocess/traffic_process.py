# -*- coding: utf-8 -*-
"""
Created on 16/6/3 00:22 2016

@author: harry sun
"""
from extend_function import *
import numpy as np


def load_traffic_data(district_dict, path):
    traffic_list = []
    temp_path1 = listdir_no_hidden(path)
    for p in temp_path1:
        temp_path2 = path+'/'+p
        with open(temp_path2) as f:
            for line in f:
                line = line.rstrip('\r\n')
                line_list = line.split('\t')
                line_list[0] = district_dict[line_list[0]]
                traffic_class = np.zeros(4, dtype='int')
                for i in range(1, len(line_list), 1):
                    if i <= 4:
                        temp = line_list[i].split(':')
                        ind = int(temp[0])
                        num = int(temp[1])
                        traffic_class[ind-1] = num
                    else:
                        time_slot = get_time_slot(line_list[i])
                temp_list = [line_list[0]] + list(traffic_class) + list(time_slot)
                traffic_list.append(temp_list)
    return traffic_list


def save_traffic_data(district_dict, path_in, path_out):
    traffic_list = load_traffic_data(district_dict, path_in)
    header = ['district_ID']
    for i in range(1, 4 + 1, 1):
        header.append('tj_class' + str(i))
    header += ['date', 'time_slot']
    write_list_to_csv(traffic_list, path_out=path_out, header=header)
