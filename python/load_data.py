import numpy as np
from preprocess.extend_function import *

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


if __name__ == '__main__':
    cluster_path = '../season_1/training_data/cluster_map/cluster_map'
    poi_path = '../season_1/training_data/poi_data/poi_data'
    poi_out_path = '../processed_data/poi_data.csv'
    write_poi(cluster_path, poi_path, poi_out_path)

