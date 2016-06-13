# -*- coding: utf-8 -*-
"""
Created on 16/6/4 21:49 2016

@author: harry sun
"""
import time
from mongo_database import connect_mongodb
import numpy as np
from extend_function import write_list_to_csv


def get_traffic_data_array_db(district_idx):
    db = connect_mongodb()
    start = time.time()
    traffic_array = np.zeros(shape=(144 * 21, 5), dtype=np.float32)
    traffic_cursor = db.traffic_data.find({'district_ID': district_idx}, no_cursor_timeout=True)

    for traffic_entry in traffic_cursor:
        date_idx = traffic_entry['date']
        time_slot_idx = traffic_entry['time_slot']
        total_idx = (date_idx-1)*144+time_slot_idx-1

        # del traffic_entry['_id']
        # del traffic_entry['district_ID']
        # del traffic_entry['time_slot']
        # del traffic_entry['date']
        traffic_array[total_idx, 0] = time_slot_idx
        p = np.asarray(traffic_entry.values())[0:4]
        traffic_array[total_idx, 1:5] = p

    #todo remove the find 0, use idx to calculate
    # fill the 0 orders time_slot
    time_slot_column = traffic_array[:, 0]
    zero_idx = np.where(time_slot_column == 0)[0]
    if zero_idx.size > 0:
        for idx in zero_idx:
            traffic_array[idx, 0] = (idx+1)%144
            if (idx+1)%144 == 0:
                traffic_array[idx, 0] = 144
            if idx%144 == 0:
                traffic_array[idx, 0] = 1

    traffic_cursor.rewind()
    end = time.time()
    print('Processed District: %d in %.2f seconds' % (district_idx, (end-start)))
    return traffic_array


def get_order_data_array_db(district_idx):
    db = connect_mongodb()
    start = time.time()
    order_array = np.zeros(shape=(144 * 21, 4), dtype=np.float32)
    order_cursor = db.order_data.find({'st_district_id': district_idx}, no_cursor_timeout=True)

    # process order data, 4 features used
    for order_entry in order_cursor:
        date_idx = order_entry['date']
        time_slot_idx = order_entry['time_slot']
        total_idx = (date_idx-1)*144+time_slot_idx-1

        order_array[total_idx, 0] = time_slot_idx
        order_array[total_idx, 1] += 1
        if len(str(order_entry['driver_id'])) < 5:
            order_array[total_idx, 2] += 1
        if order_entry['ed_district_id'] == 0:
            order_array[total_idx, 3] += 1

    #todo remove the find 0, use idx to calculate
    # fill the 0 orders time_slot
    time_slot_column = order_array[:, 0]
    zero_idx = np.where(time_slot_column == 0)[0]
    if zero_idx.size > 0:
        for idx in zero_idx:
            order_array[idx, 0] = (idx+1)%144
            if (idx+1)%144 == 0:
                order_array[idx, 0] = 144
            if idx%144 == 0:
                order_array[idx, 0] = 1

    order_cursor.rewind()
    end = time.time()
    print('Processed District: %d in %.2f seconds' % (district_idx, (end-start)))
    return order_array


def get_weather_data_array_db():
    db = connect_mongodb()
    weather_array = np.zeros(shape=(144 * 21, 4), dtype=np.float32)
    weather_cursor = db.weather_data.find({})
    for weather_entry in weather_cursor:
        date_idx = weather_entry['date']
        time_slot_idx = weather_entry['time_slot']
        total_idx = (date_idx - 1) * 144 + time_slot_idx - 1
        if np.sum(weather_array[total_idx]) == 0:
            weather_array[total_idx, 0] = time_slot_idx
            #todo use one line to assign the value
            weather_array[total_idx, 1] = weather_entry['Weather']
            weather_array[total_idx, 2] = weather_entry['temprature']
            weather_array[total_idx, 3] = weather_entry['PM25']
        else:
            weather_array[total_idx, 1] = (weather_array[total_idx, 1]+weather_entry['Weather'])/2.0
            weather_array[total_idx, 2] = (weather_array[total_idx, 2]+weather_entry['temprature'])/2.0
            weather_array[total_idx, 3] = (weather_array[total_idx, 3]+weather_entry['PM25'])/2.0

    # find the 0 time_slot
    time_slot_column = weather_array[:, 0]
    zero_idx = np.where(time_slot_column == 0)[0]
    if zero_idx.size > 0:
        for idx in zero_idx:
            # fill the zeros of the weather data
            count = 0
            temp_array = np.zeros(shape=(1, 3))
            for windows_idx in range(1, 6+1, 1):
                temp_idx = idx-4+windows_idx
                if np.sum(weather_array[temp_idx, 1:4]) != 0:
                    count += 1
                    temp_array += weather_array[temp_idx, 1:4]
            array = temp_array/float(count)
            weather_array[idx, 1:4] = array
            # fill the 0 time slot
            weather_array[idx, 0] = (idx + 1) % 144
            if (idx+1)%144 == 0:
                weather_array[idx, 0] = 144
            if idx%144 == 0:
                weather_array[idx, 0] = 1
    return weather_array


# def get_
if __name__ == '__main__':
    p = get_traffic_data_array_db(54)
    weather_array = get_weather_data_array_db()

    for district_id in range(1, 66+1, 1):
        order_array = get_order_data_array_db(district_id)
        order_list = order_array.tolist()
        order_path = '../../processed_data/train/district_' + str(district_id) + '_training_data.csv'
        write_list_to_csv(order_list, order_path)

    #
