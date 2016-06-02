import os
import pandas as pd



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
    table = pd.read_table(path, names=['district hash', 'district_id'])
    array = table.get_values()
    district_dict = {array[i][0]: array[i][1] for i in range(0, len(array), 1)}
    return district_dict

def get_time_slot(time):
    time = time.replace(' ', ':')
    time = time.replace('-', ':')
    time_list = time.split(':')
    temp1 = time_list[1]+time_list[2]
    t1 = int(time_list[3])
    t2 = int(time_list[4])
    temp2 = (t1 * 6 + t2 / 10) + 1
    time_slot = [temp1, temp2]
    return time_slot


if __name__ == '__main__':
    get_time_slot(time='2016-01-01 23:30:22')
