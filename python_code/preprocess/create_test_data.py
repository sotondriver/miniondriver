# -*- coding: utf-8 -*-
"""
Created on 16/6/14 19:55 2016

@author: harry sun
"""
import time
from mongo_database import connect_mongodb
import numpy as np
from extend_function import write_list_to_csv

date_window = [23, 25, 27, 29, 31]
predict_time_slot_window = [43, 44, 45, 55, 56, 57, 67, 68, 69, 79, 80, 81, 91, 92, 93, 103, 104,
                            105, 115, 116, 117, 127, 128, 129, 139, 140, 141]

def  get_traffic_data_array_db(district_idx):
    db = connect_mongodb()
    start = time.time()
    traffic_array = np.zeros(shape=(27 * 5, 5), dtype=np.float32)
    traffic_cursor = db.get_collection('test_traffic_data').find({'district_ID': district_idx}, no_cursor_timeout=True)

    for traffic_entry in traffic_cursor:
        date_idx = traffic_entry['date']
        time_slot_idx = traffic_entry['time_slot']
        total_idx = get_idx(date_idx, time_slot_idx)

        # del traffic_entry['_id']
        # del traffic_entry['district_ID']
        # del traffic_entry['time_slot']
        # del traffic_entry['date']
        traffic_array[total_idx, 0] = time_slot_idx
        p = np.asarray(traffic_entry.values())[0:4]
        traffic_array[total_idx, 1:5] = p

    # fill the 0 orders time_slot
    time_slot_column = traffic_array[:, 0]
    zero_idx = np.where(time_slot_column == 0)[0]
    if zero_idx.size > 0:
        print('How many zeros: %d' % (zero_idx.size))
        for idx in zero_idx:
            traffic_array[idx, 0] = predict_time_slot_window[(idx) % 27]

    traffic_cursor.rewind()
    end = time.time()
    print('Processed Traffic in District: %d in %.2f seconds' % (district_idx, (end - start)))
    return traffic_array


def get_weather_data_array_db(name):
    db = connect_mongodb()
    weather_array = np.zeros(shape=(27*5, 4), dtype=np.float32)
    weather_cursor = db.get_collection(name).find({})
    for weather_entry in weather_cursor:
        date_idx = weather_entry['date']
        time_slot_idx = weather_entry['time_slot']
        total_idx = get_idx(date_idx, time_slot_idx)
        if np.sum(weather_array[total_idx]) == 0:
            weather_array[total_idx, 0] = time_slot_idx
            # todo use one line to assign the value
            weather_array[total_idx, 1] = float(weather_entry['Weather'])
            weather_array[total_idx, 2] = float(weather_entry['temprature'])
            weather_array[total_idx, 3] = float(weather_entry['PM25'])
        else:
            weather_array[total_idx, 1] = float(weather_array[total_idx, 1] + weather_entry['Weather']) / 2.0
            weather_array[total_idx, 2] = float(weather_array[total_idx, 2] + weather_entry['temprature']) / 2.0
            weather_array[total_idx, 3] = float(weather_array[total_idx, 3] + weather_entry['PM25']) / 2.0

    # find the 0 time_slot
    time_slot_column = weather_array[:, 0]
    zero_idx = np.where(time_slot_column == 0)[0]
    if zero_idx.size > 0:
        for idx in zero_idx:
            weather_array[idx, 0] = predict_time_slot_window[(idx) % 27]
    return weather_array


def get_idx(date, time_slot):
    idx = None

    if date in date_window:
        date_idx = date_window.index(date)
        if time_slot in predict_time_slot_window:
            temp_idx = predict_time_slot_window.index(time_slot)
            idx = (date_idx * 27) + temp_idx
    return idx

if __name__ == '__main__':
    st = time.time()
    #
    # weather_array = get_weather_data_array_db('test_weather_data')
    # weather_list = weather_array.tolist()
    # weather_path = '../../processed_data/test/weather_data.csv'
    # write_list_to_csv(weather_list, weather_path)

    for district_id in range(1, 66 + 1, 1):
        # order_array = get_order_data_array_db(district_id, 'test_order_data')
        traffic_array = get_traffic_data_array_db(district_id)

        # order_list = order_array.tolist()
        traffic_list = traffic_array.tolist()

        order_path = '../../processed_data/test/D' + str(district_id) + '_order_data.csv'
        traffic_path = '../../processed_data/test/D' + str(district_id) + '_traffic_data.csv'

        # write_list_to_csv(order_list, order_path)
        write_list_to_csv(traffic_list, traffic_path)

        print('==================================================')
    ed = time.time()
    print('Overall time: %.2f Minutes' % ((ed-st)/60))