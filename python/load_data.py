import os
import pandas as pd


class order(object):

    def __init__(self):
        self._order_id = order_id
        self._driver_id = driver_id
        self._passenger_id = passenger_id
        self._district_st_hash = district_st_hash
        self._district_ed_hash = district_ed_hash
        self._price = price
        self._time = time
        self._weather = weather
        self._temp = temp
        self._pm25 = pm25




def listdir_no_hidden(path):
    list1 = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            list1.append(f)
    return list1

