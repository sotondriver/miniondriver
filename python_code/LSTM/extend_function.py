# -*- coding: utf-8 -*-
"""
Created on 16/6/8 14:52 2016

@author: harry sun
"""
import pandas as pd

def write_list_to_csv(list1, path_out, header=False):
    temp = pd.DataFrame(list1)
    temp.to_csv(path_out, index=False, header=header)