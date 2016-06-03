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


def save_individual_csv():
    parent_in_path = '../../training_data/'
    parent_out_path = '../../processed_data/'

    # the path of all the necessary file location
    cluster_path = parent_in_path + 'cluster_map/cluster_map'

    poi_in_path = parent_in_path + 'poi_data/poi_data'
    poi_out_path = parent_out_path + 'poi_data.csv'

    traffic_in_path = parent_in_path + 'traffic_data'
    traffic_out_path = parent_out_path + 'traffic_data.csv'

    weather_in_path = parent_in_path + 'weather_data'
    weather_out_path = parent_out_path + 'weather_data.csv'

    order_in_path = parent_in_path + 'order_data'
    order_out_path = parent_out_path + 'order_data.csv'

    # the hash dictionary for the district map ID
    district_dict = load_cluster_map(cluster_path)

    # save those csv file for the data cleaness
    save_traffic_data(district_dict, traffic_in_path, traffic_out_path)
    save_poi_data(district_dict, poi_in_path, poi_out_path)
    save_weather_data(weather_in_path, weather_out_path)
    save_order_data(order_in_path, order_out_path, cluster_path)

if __name__ == '__main__':
    save_individual_csv()
