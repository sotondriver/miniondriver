# -*- coding: utf-8 -*-
"""
Created on 16/6/3 00:22 2016

@author: harry sun
"""
from extend_function import write_list_to_csv
import numpy as np


def load_poi(district_dict, path):
    poi_list = []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            line_list = line.split('\t')
            line_list[0] = district_dict[line_list[0]]
            poi_1class = np.zeros(25, dtype='int')
            for i in range(1, len(line_list), 1):
                temp = line_list[i].split('#')
                temp2 = temp[-1].split(':')
                splitted = temp[0:-1] + temp2
                poi_1class_ind = int(splitted[0])
                poi_1class_num = int(splitted[-1])
                poi_1class[poi_1class_ind-1] += poi_1class_num
            poi_list.append([line_list[0]] + list(poi_1class))
    return poi_list


def save_poi_data(district_dict, path_in, path_out):
    poi_list = load_poi(district_dict, path_in)
    header = ['district_ID']
    for i in range(1, 25+1, 1):
        header.append('class'+str(i))
    write_list_to_csv(poi_list, path_out=path_out, header=header)