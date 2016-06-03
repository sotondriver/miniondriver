# -*- coding: utf-8 -*-
"""
Created on Wed Jun 01 00:11:48 2016

@author: abulin
"""


import os
import pandas as pd
def MakeWeatherTable(root):
    Table = []
    Root = []
    root = root
    for path, subdirs, files in os.walk(root):
        for name in files:
            finalPath = './' + path + '/' + name
            Root.append(finalPath)
    for r in Root:
        with open(r, 'r') as a:
            data = a.readlines()
            for line in data:
                words = line.split()
                d = words[0][5:10]
                d = d.split('-')
                day = d[0] + d[1]
                words[0] = day
                t = words[1][0:5]
                t = t.split(':')
                t1 = int(t[0])
                t2 = int(t[1])
                timeP = (t1 * 6 + t2 / 10) + 1
                words[1] = str(timeP)
                w = words[2]
                words[2] = w
                Table.append(words)
    return Table

def save_weather_data(path_in, path_out):
    Table = MakeWeatherTable(path_in)
    table = pd.DataFrame(Table)
    header = ['date','time-slot','weather','temprature','PM2.5']
    table.to_csv(path_out, index = False, header = header)