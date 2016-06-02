import os
import pandas as pd
from itertools import izip



# class order(object):
#
#     def __init__(self):
#         self._order_id = order_id
#         self._driver_id = driver_id
#         self._passenger_id = passenger_id
#         self._district_st_hash = district_st_hash
#         self._district_ed_hash = district_ed_hash
#         self._price = price
#         self._time = time
#         self._weather = weather
#         self._temp = temp
#         self._pm25 = pm25


def listdir_no_hidden(path):
    list1 = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            list1.append(f)
    return list1

def load_cluster_map(path):
    global district_dict
    table = pd.read_table(path,names=['district hash','district_id'])
    array = table.get_values()
    district_dict = {array[i][0]: array[i][1] for i in range(0, len(array), 1)}
    return district_dict

def load_poi(path, district_dict):

    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            linedata = line.split('\t')
            linedata[0] = district_dict[linedata[0]]

if __name__ == '__main__':
    p1 = '../season_1/training_data/cluster_map/cluster_map'
    p2 = '../season_1/training_data/poi_data/poi_data'
    district_dict = load_cluster_map(p1)
    load_poi(p2, district_dict)
