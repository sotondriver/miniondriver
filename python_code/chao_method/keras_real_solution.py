import matplotlib.pyplot as plt  
import numpy as np  
from sklearn import linear_model  

from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation  
from keras.optimizers import SGD

from keras.callbacks import Callback
  
from numpy import genfromtxt, savetxt
TRAIN_RATIO = 0.8
  
data_list = []
for i in range(1,BASE_POI_NUM+1):
    select_features = data_[:,[0,i,i+BASE_POI_NUM,i+2*BASE_POI_NUM]] # select partial features
#    select_features = data_
    select_label_idx = np.where(label_[:,i-1]>0)[0] # select nonnegative datas
    data_i = select_features[select_label_idx,:]
    label_i = label_[select_label_idx,i-1]
    data_list.append((data_i,label_i))

class LossHistory(Callback):  
    def on_train_begin(self, logs={}):  
        self.losses = []  
    def on_batch_end(self, batch, logs={}):  
        self.losses.append(logs.get('loss'))  
# training
model_list = None
if not 'model_list' in dir() or model_list == None:
    sum_ = 0
    num_ = 0
    model_list = []
#    for i in range(len(data_list)):
    for i in range(11,12):    
        print 'calcu: %d' %i
        (my_data,my_label) = data_list[i]
        train_num = int(np.floor(len(my_data)*TRAIN_RATIO))
        train_data = my_data[0:train_num]
        test_data = my_data[train_num:]
        train_label = my_label[0:train_num]
        test_label = my_label[train_num:]
        activator = 'linear'
#        activator = 'sigmoid'
        model = Sequential()
        model.add(Dense(20, input_dim=train_data.shape[1],activation=activator))
        model.add(Dropout(0.5))
        model.add(Dense(20, activation=activator))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.compile(loss='mape', optimizer='adam')
        history = LossHistory()
        model.fit(train_data, train_label, verbose=False, nb_epoch=80, batch_size=16, callbacks=[history])
        model_list.append(model)
        predict_result = model.predict(test_data).reshape(len(test_label))
        a = np.abs(test_label-predict_result)
        for j in range(len(predict_result)):
            if test_label[j] != 0:
                sum_ = sum_ + a[j]/test_label[j]
                num_ = num_ + 1
    print 'mape: %f\n' %(sum_/num_)

## predict
#OUTPUT_TEST_FEATURE_PATH_FILEPATH = '/Users/channerduan/Desktop/didi_test_data.csv'
#OUTPUT_RESULT_FILEPATH = '/Users/channerduan/Desktop/didi.csv'
#time_str = '2016-01-22-46,2016-01-22-58,2016-01-22-70,2016-01-22-82,2016-01-22-94,2016-01-22-106,2016-01-22-118,2016-01-22-130,2016-01-22-142,2016-01-24-58,2016-01-24-70,2016-01-24-82,2016-01-24-94,2016-01-24-106,2016-01-24-118,2016-01-24-130,2016-01-24-142,2016-01-26-46,2016-01-26-58,2016-01-26-70,2016-01-26-82,2016-01-26-94,2016-01-26-106,2016-01-26-118,2016-01-26-130,2016-01-26-142,2016-01-28-58,2016-01-28-70,2016-01-28-82,2016-01-28-94,2016-01-28-106,2016-01-28-118,2016-01-28-130,2016-01-28-142,2016-01-30-46,2016-01-30-58,2016-01-30-70,2016-01-30-82,2016-01-30-94,2016-01-30-106,2016-01-30-118,2016-01-30-130,2016-01-30-142'
#list_slot_str = time_str.split(',')
#test_data = genfromtxt(open(OUTPUT_TEST_FEATURE_PATH_FILEPATH,'r'), delimiter=',', dtype='int')
#slot_num = test_data.shape[0]
#f_out = open(OUTPUT_RESULT_FILEPATH,'w')
#res_list = []
#for i in range(1,BASE_POI_NUM+1):
#    select_ = test_data[:,[0,i,i+BASE_POI_NUM,i+2*BASE_POI_NUM]]
#    res_list.append(model_list[i-1].predict(select_))
#for i in range(slot_num):
#    for j in range(BASE_POI_NUM):
#        f_out.write('%d,%s,%f\n' %(j+1,list_slot_str[i],res_list[j][i,0]))
#f_out.close()






