# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 15:47:39 2016

@author: yx
"""

import pandas as pd
import numpy as np
import os

root ='training_data/order_data'
Root = []
order_datas = []
for path, subdirs, files in os.walk(root):
        for name in files:
            finalPath = './' + path + '/' + name
            Root.append(finalPath)
    
cluster_map = pd.read_table('training_data/cluster_map/cluster_map', names=['district_hash', 'district_id'])
order_data = pd.read_table('training_data/order_data/order_data_2016-01-04', names=['order_id', 'driver_id','passenger_id','start_district_hash','dest_district_hash','Price','Time'])

for i in range(len(Root)):
    print i
    order_datas.append(pd.read_table(Root[i], names=['order_id', 'driver_id','passenger_id','start_district_hash','dest_district_hash','Price','Time']))


def hash_table(order_data):
    start_areas = order_data['start_district_hash']
    start_areas = start_areas.get_values()
    dest_areas = order_data['dest_district_hash']
    dest_areas = dest_areas.get_values()
    values = cluster_map['district_id']
    keys = cluster_map['district_hash']
    values = values.get_values()
    keys = keys.get_values()
    dict = {k:v for k, v in zip(keys, values)}

    for i in range(len(order_data['start_district_hash'])):
#        if dict.has_key(dest_areas[i]) == True:
        start_areas[i] = dict[start_areas[i]]
#        else:
#            start_areas[i] = 0
        
    for j in range(len(order_data['dest_district_hash'])):
        if dict.has_key(dest_areas[j]) == True:
            dest_areas[j] = dict[dest_areas[j]]
        else:
            dest_areas[j] = 0
    return start_areas, dest_areas
    
def get_time(order_data) :
    order_time = order_data['Time']
    int_time = []#np.empty([len(order_time),2])
    time_day = []
    order_time = order_time.get_values()
    for i in range(len(order_time)):        
        int_time.append( order_time[i].split())#filter(lambda x:x.isdigit(),order_time[i])
        time_day.append(int_time[i][1].split(":"))
    return time_day    

def calculate_time(time_day):
    s = np.empty([len(time_day)])
    for i in range(len(time_day)):
        s[i] = (int(time_day[i][0]) + 1)*(int(time_day[i][1])/10 + 1)
    return s

def calculate_gap(order_data):
    gap_one = order_data['driver_id']
    gap_one = gap_one.get_values()
    for i in range(len(gap_one)):
        if str(gap_one[i]) == 'nan':
            gap_one[i] = 0
        else:
            gap_one[i] = 1
    return gap_one
    
def new_table(order_data,day):
    start_areas,dest_areas = hash_table(order_data)    
    start_areas = start_areas.astype(int)  
    dest_areas = dest_areas.astype(int)

    order_data['start_district_hash'] = start_areas
    order_data['dest_district_hash'] = dest_areas

    time_day = get_time(order_data)
    #int_time = int_time.astype(int)
    s = calculate_time(time_day)    
    order_data['Time'] = s
    gap_one = calculate_gap(order_data)
    order_data['driver_id'] = gap_one
    
    del order_data['order_id']
    del order_data['passenger_id']
    order_data = order_data.as_matrix()
    
    order_number = []
    time_number = []
    area_number = []
    gap_number = []
    day_number = []
    for i in range(1,145):
        area = order_data[order_data[:,4] == i]
        for j in range(1,67):
            area_i = area[area[:,1] == j]
            order_number.append(len(area_i))
            time_number.append(i)
            area_number.append(j)
            gap_number.append(len(area_i[area_i[:,0] == 0]))
            day_number.append(day+1)
    new_table = np.array([day_number,order_number,time_number,area_number,gap_number])
    new_table = new_table.transpose()
    return new_table

def write_list_to_csv(list1, outpath, header):
    temp = pd.DataFrame(list1)
    temp.to_csv(outpath, index=False, header=header)
  
#new_tables = new_table(order_data,0)  
new_tables =  []
for i in range(len(order_datas)):
    print i
    new_tables.extend(new_table(order_datas[i],i)) 
header = ['date','total_order_number','time_slot','area_number','gap']

write_list_to_csv(new_tables,outpath='/Users/yx/Desktop/season_1/code/order_data.csv',header=header )
