import os
import pandas as pd


def listdir_no_hidden(path):
    list1 = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            list1.append(f)
    return list1


def write_list_to_csv(list1, outpath):
    temp = pd.DataFrame(list1)
    temp.to_csv(outpath, index=False, header=False)

def load_cluster_map(path):
    table = pd.read_table(path,names=['district hash','district_id'])
    array = table.get_values()
    district_dict = {array[i][0]: array[i][1] for i in range(0, len(array), 1)}
    return district_dict