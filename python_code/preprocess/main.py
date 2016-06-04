# -*- coding: utf-8 -*-
"""
Created on 16/6/3 00:22 2016

@author: harry sun
"""
from extend_function import *
from poi_process import save_poi_data
from traffic_process import save_traffic_data
from weather_process import save_weather_data
from order_process import save_order_data

PARENT_IN_PATH = '../../training_data/'
PARENT_OUT_PATH = '../../processed_data/'

# the path of all the necessary file location
CLUSTER_PATH = PARENT_IN_PATH + 'cluster_map/cluster_map'

POI_IN_PATH = PARENT_IN_PATH + 'poi_data/poi_data'
POI_OUT_PATH = PARENT_OUT_PATH + 'poi_data.csv'

TRAFFIC_IN_PATH = PARENT_IN_PATH + 'traffic_data'
TRAFFIC_OUT_PATH = PARENT_OUT_PATH + 'traffic_data.csv'

WEATHER_IN_PATH = PARENT_IN_PATH + 'weather_data'
WEATHER_OUT_PATH = PARENT_OUT_PATH + 'weather_data.csv'

ORDER_IN_PATH = PARENT_IN_PATH + 'order_data'
ORDER_OUT_PATH = PARENT_OUT_PATH + 'order_data.csv'

def save_individual_csv():
    # the hash dictionary for the district map ID
    district_dict = load_cluster_map(CLUSTER_PATH)

    # save those csv file for the data cleaness
    save_traffic_data(district_dict, TRAFFIC_IN_PATH, TRAFFIC_OUT_PATH)
    save_poi_data(district_dict, POI_IN_PATH, POI_OUT_PATH)
    save_weather_data(WEATHER_IN_PATH, WEATHER_OUT_PATH)
    save_order_data(ORDER_IN_PATH, ORDER_OUT_PATH, CLUSTER_PATH)

if __name__ == '__main__':
    save_individual_csv()
