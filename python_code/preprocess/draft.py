# -*- coding: utf-8 -*-
"""
Created on 16/6/6 09:54 2016

@author: harry sun
"""
import pymongo
from extend_function import *
import main as m
import numpy as np

client = pymongo.MongoClient("localhost", 27017)
db = client.dididata

temp_path1 = listdir_no_hidden(m.ORDER_IN_PATH)
gap = []
for p in temp_path1:
    for id in range(1, 66 + 1, 1):
        for time in range(1, 144 +1, 1):
            t = db.get_collection(p).find({'st_district_id':id,'time_slot':time,'driver_id':np.nan}).count()
            gap.append(t)
            print(t)

# idx = int(entry['st_district_id'] * entry['time_slot']) - 1
# order_table[idx, 0] = int(p[-1])
# order_table[idx, 1] = entry['st_district_id']
# order_table[idx, 2] = entry['time_slot']
# order_table[idx, 3] += 1
# if entry['ed_district_id'] == 0:
#     order_table[idx, 4] += 1
# if str(entry['driver_id']) == 'nan':
#     order_table[idx, 5] += 1
# order_list = list(order_table)