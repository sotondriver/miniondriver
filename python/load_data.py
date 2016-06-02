# -*- coding: utf-8 -*-
"""
Created on 16/6/3 00:22 2016

@author: harry sun
"""
from preprocess.extend_function import *
from preprocess.poi_process import save_poi_data
from preprocess.traffic_process import save_traffic_data

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


if __name__ == '__main__':
    cluster_path = '../season_1/training_data/cluster_map/cluster_map'

    poi_in_path = '../season_1/training_data/poi_data/poi_data'
    poi_out_path = '../processed_data/poi_data.csv'

    traffic_in_path = '../season_1/training_data/traffic_data'
    traffic_out_path = '../processed_data/traffic_data.csv'

    district_dict = load_cluster_map(cluster_path)

    save_traffic_data(district_dict, traffic_in_path, traffic_out_path)
    save_poi_data(district_dict, poi_in_path, poi_out_path)
