# -*- coding: utf-8 -*-
"""
Created on 16/6/4 21:49 2016

@author: harry sun
"""
import time
from mongo_database import connect_mongodb
import numpy as np
from extend_function import write_list_to_csv

def get_train_data_array_db(district_idx):
    db = connect_mongodb()
    start = time.time()
    total_array = np.zeros(shape=(144 * 21, 4), dtype=np.float32)
    order_db = []
    traffic_db = []

    poi_db = db.poi_data.find_one({'district_id':district_idx})
    order_cursor = db.order_data.find({'st_district_id': district_idx}, no_cursor_timeout=True)
    traffic_cursor = db.traffic_data.find({'district_ID': district_idx}, no_cursor_timeout=True)

    # for poi data
    del poi_db['_id']
    del poi_db['district_id']
    poi_list = list(poi_db.values())

    # process order data, 4 features used
    for order_entry in order_cursor:
        date_idx = order_entry['date']
        time_slot_idx = order_entry['time_slot']
        total_idx = (date_idx-1)*144+time_slot_idx-1

        total_array[total_idx, 0] = time_slot_idx
        total_array[total_idx, 1] += 1
        if len(str(order_entry['driver_id'])) < 5:
            total_array[total_idx, 2] += 1
        if order_entry['ed_district_id'] == 0:
            total_array[total_idx, 3] += 1
    # fill the 0 orders time_slot
    time_slot_column = total_array[:, 0]
    zero_idx = np.where(time_slot_column == 0)[0]
    if zero_idx.size > 0:
        for idx in zero_idx:
            total_array[idx, 0] = (idx+1)%144
            if (idx+1)%144 == 0:
                total_array[idx, 0] = 144
            if idx%144 == 0:
                total_array[idx, 0] = 1

    # for date_idx in range(1, 21, 1):
    #     for time_slot_idx in range(1, 144, 1):
    #         # for order data
    #         gap_count = 0
    #         no_ed_district_id_count = 0
    #         order_count = 0
    #
    #         for order_entry in order_db:
    #             if (order_entry['date'] == date_idx) & (order_entry['time_slot'] == time_slot_idx):
    #                 order_count += 1
    #                 if len(str(order_entry['driver_id'])) < 5:
    #                     gap_count += 1
    #                 if order_entry['ed_district_id'] == 0:
    #                     no_ed_district_id_count += 1
    #
    #         # for traffic data
    #         temp_traffic_array = np.zeros(4, dtype='int')
    #         for traffic_entry in traffic_db:
    #             if traffic_entry.has_key('date'):
    #                 if (traffic_entry['date'] == date_idx) & (traffic_entry['time_slot'] == time_slot_idx):
    #                     temp_traffic_entry = traffic_entry
    #                     del temp_traffic_entry['_id']
    #                     del temp_traffic_entry['district_ID']
    #                     del temp_traffic_entry['time_slot']
    #                     del temp_traffic_entry['date']
    #                     temp_traffic_array = np.vstack([temp_traffic_array, list(temp_traffic_entry.values())])
    #         t = temp_traffic_array.shape
    #         if len(t) > 1 :
    #             traffic_list = np.mean(temp_traffic_array, axis=0).tolist()
    #         else:
    #             traffic_list = temp_traffic_array.tolist()
    #
    #         # weather data
    #         temp_weather_array = np.zeros(3, dtype='int')
    #         weather_db = db.weather_data.find_one({'date': date_idx,'time_slot': time_slot_idx})
    #         if weather_db is not None:
    #             del weather_db['_id']
    #             del weather_db['date']
    #             del weather_db['time_slot']
    #             weather_list = list(weather_db.values())
    #         else:
    #             weather_list = temp_weather_array.tolist()
    #
    #         entry_list = [district_idx, date_idx, time_slot_idx, order_count, no_ed_district_id_count]
    #         entry_list += poi_list
    #         entry_list += traffic_list
    #         entry_list += weather_list
    #         entry_list.append(gap_count)
    #         count += 1
    #         if (count%10000) == 0:
    #             print(count)
    #         total_list.append(entry_list)
    order_cursor.rewind()
    traffic_cursor.rewind()
    end = time.time()
    print('Processed District: %d in %.2f seconds' % (district_idx, (end-start)))
    return total_array