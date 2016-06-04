# -*- coding: utf-8 -*-
"""
Created on 16/6/3 23:49 2016

@author: harry sun
"""
import pymongo
import numpy as np
from extend_function import *
from main import CLUSTER_PATH, ORDER_IN_PATH, WEATHER_IN_PATH

ORDER_NAMES = ['order_id', 'driver_id', 'passenger_id', 'start_district_hash',
                'dest_district_hash', 'Price', 'Time']
WEATHER_NAMES = ['Time', 'Weather', 'temprature', 'PM25']


def save_data_into_mongodb(path, names):

    temp_path1 = listdir_no_hidden(path)
    for p in temp_path1:
        temp_path2 = path+'/'+p
        order_dict = pd.read_table(temp_path2, names=names).to_dict(orient='records')
        db.raw_data.insert(order_dict)
        print(p)


if __name__ == '__main__':
    client = pymongo.MongoClient("localhost", 27017)
    db = client.dididata

    # district_dict = load_cluster_map(CLUSTER_PATH)
    # db.cluster_map.insert_one(district_dict)
    # save_data_into_mongodb(WEATHER_IN_PATH, WEATHER_NAMES)
    a = db.raw_data.distinct("order_id").length
    print(a)

    pass