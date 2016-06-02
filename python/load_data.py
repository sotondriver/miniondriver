import numpy as np
import pandas as pd
from preprocess.extend_function import *

# class order(object):
#
#     def __init__(self):
#         self._order_id = order_id
#         self._driver_id = driver_id
#         self._passenger_id = passenger_id
#         self._district_st_hash = district_st_hash
#         self._district_ed_hash = district_ed_hash
#         self._price = price
#         self._time = time
#         self._weather = weather
#         self._temp = temp
#         self._pm25 = pm25


def load_poi(path, district_dict):
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


if __name__ == '__main__':
    p1 = '../season_1/training_data/cluster_map/cluster_map'
    p2 = '../season_1/training_data/poi_data/poi_data'
    district_dict = load_cluster_map(p1)
    poi_list = load_poi(p2, district_dict)
    write_list_to_csv(poi_list, outpath='../processed_data/poi_data.csv')


