# -*- coding: utf-8 -*-
"""
Created on 16/6/3 23:49 2016

@author: harry sun
"""
import csv

import pymongo
import numpy as np
from extend_function import *
import main as m


def connect_mongodb():
    client = pymongo.MongoClient("localhost", 27017)
    db = client.dididata
    return db

def save_data_into_mongodb(db, path, names, collection):
    temp_path1 = listdir_no_hidden(path)
    for p in temp_path1:
        temp_path2 = path+'/'+p
        temp_dict = pd.read_table(temp_path2, names=names).to_dict(orient='records')
        db.get_collection(collection).insert(temp_dict)
        print('Insert file: '+p)


def save_csv_into_mongodb(db, path, collection):
    temp_dict1 = pd.read_table(path, header=0, delimiter=',')
    temp_dict = temp_dict1.to_dict(orient='records')
    db.get_collection(collection).insert(temp_dict)


if __name__ == '__main__':
    db = connect_mongodb()

    # save the raw data into mongodb "dididata"
    # save_data_into_mongodb(db, m.WEATHER_IN_PATH, m.WEATHER_NAMES, collection='weather_data')
    # save_data_into_mongodb(db, m.ORDER_IN_PATH, m.ORDER_NAMES, collection='order_data')
    # save_csv_into_mongodb(db, m.TRAFFIC_OUT_PATH, collection='traffic_data')
    save_csv_into_mongodb(db, m.POI_OUT_PATH, collection='poi_data')
    pass