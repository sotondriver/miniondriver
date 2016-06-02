import os
import pandas as pd
import numpy as np


def listdir_no_hidden(path):
    list1 = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            list1.append(f)
    return list1


def write_list_to_csv(list1, outpath, header=False):
    temp = pd.DataFrame(list1)
    temp.to_csv(outpath, index=False, header=header)


def load_cluster_map(path):
    table = pd.read_table(path,names=['district hash','district_id'])
    array = table.get_values()
    district_dict = {array[i][0]: array[i][1] for i in range(0, len(array), 1)}
    return district_dict


def load_poi(path, district_dict):
    poi_list = []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            line_list = line.split('\t')
            line_list[0] = district_dict[line_list[0]]
            poi_1class = np.zeros(25, dtype='int')
            for i in range(1, len(line_list), 1):
                temp = line_list[i].split('#')
                temp2 = temp[-1].split(':')
                splitted = temp[0:-1] + temp2
                poi_1class_ind = int(splitted[0])
                poi_1class_num = int(splitted[-1])
                poi_1class[poi_1class_ind-1] += poi_1class_num
            poi_list.append([line_list[0]] + list(poi_1class))
    return poi_list

def write_poi(p1, p2, p3):
    district_dict = load_cluster_map(p1)
    poi_list = load_poi(p2, district_dict)
    header = ['district_ID']
    for i in range(1, 25+1, 1):
        header.append('class'+str(i))
    write_list_to_csv(poi_list, outpath=p3, header=header)
