# -*- coding: utf-8 -*-
"""
Created on 16/6/3 23:49 2016

@author: harry sun
"""
import csv
from operator import itemgetter

import numpy as np

import pymongo
from extend_function import *
import main as m


def connect_mongodb():
    client = pymongo.MongoClient("localhost", 27017)
    db = client.didi_clean_data
    return db


def save_data_into_mongodb(db, path, names, collection, clean=False):
    district_dict = load_cluster_map(m.CLUSTER_PATH)
    temp_path1 = listdir_no_hidden(path)
    for p in temp_path1:
        temp_path2 = path+'/'+p
        temp_dict = pd.read_table(temp_path2, names=names).to_dict(orient='records')
        temp_dict1 = temp_dict
        if clean == True:
            temp_dict1 = clean_temp_dict(temp_dict, district_dict)
        for id in range(1, 66 + 1, 1):
            value_list = map(itemgetter('st_district_id'), temp_dict1)
            idx = np.where(np.array(value_list) == id)[0]
            temp = itemgetter(*idx)(temp_dict1)
            db.get_collection(p+'district'+str(id)).insert(temp)
            print('district'+str(id))
        print('Insert file: '+p)


def save_csv_into_mongodb(db, path, collection):
    temp_dict1 = pd.read_table(path, header=0, delimiter=',')
    temp_dict = temp_dict1.to_dict(orient='records')
    db.get_collection(collection).insert(temp_dict)


def clean_temp_dict(temp_dict, district_dict):
    count = 0
    for entry in temp_dict:
        entry['start_district_hash'] = district_dict[entry['start_district_hash']]
        entry['st_district_id'] = entry.pop('start_district_hash')
        temp_time = entry['Time']
        time_list = get_time_slot(temp_time)
        entry['date'] = time_list[0]
        entry['time_slot'] = time_list[1]
        del entry['Time']
        id_flag = district_dict.get(entry['dest_district_hash'])
        if id_flag is not None:
            entry['dest_district_hash'] = district_dict[entry['dest_district_hash']]
        elif id_flag is None:
            entry['dest_district_hash'] = 0
        entry['ed_district_id'] = entry.pop('dest_district_hash')
        count += 1
        if count%100000 == 0:
            print(count)
    return temp_dict


if __name__ == '__main__':
    dididata = connect_mongodb()

    # save the raw data into mongodb "dididata"
    # save_data_into_mongodb(dididata, m.CLUSTER_PATH, m.CLUSTER_NAMES, collection='cluster_map')
    # save_data_into_mongodb(dididata, m.WEATHER_IN_PATH, m.WEATHER_NAMES, collection='weather_data')
    save_data_into_mongodb(dididata, m.ORDER_IN_PATH, m.ORDER_NAMES, collection='order_data', clean=True)
    # save_csv_into_mongodb(dididata, m.TRAFFIC_OUT_PATH, collection='traffic_data')
    # save_csv_into_mongodb(dididata, m.POI_OUT_PATH, collection='poi_data')

    pass