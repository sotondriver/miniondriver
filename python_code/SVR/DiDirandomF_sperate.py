# -*- coding: utf-8 -*-
"""
Created on Mon Jun 06 02:15:40 2016

@author: abulin
"""

from __future__ import division 
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import linear_model
import copy
import pandas as pd
from Kmean_dis import KmeansPOI
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from collections import Counter

sample = []
samples = []
samples_trafic = []
samples_train = []
samples_test = []
responses = []
clf = []
coef = []
results = []
Results = []
weights = []
over_04_list = []
over_04_value = []
equal = []
equals = []
equal_value = []
equal_values = []
sum_ = 0
num_ = 0
tt = 2369
LF = 1000

with open('didi_train_data.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        row = row[0].split(',')
        row = [int(i) for i in row ]    
        sample.append(row)

#with open('didi_traffic.csv', 'rb') as csvfile:
#    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#    for trafic in spamreader:
#        trafic = trafic[0].split(',')
#        trafic = [int(i)/100 for i in trafic ]    
#        samples_trafic.append(trafic)

with open('didi_train_label.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for lable in spamreader:
        lable = lable[0].split(',')
        lable = [ int(i) for i in lable]
        responses.append(lable)

with open('best_weights.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for lable in spamreader:
        lable = lable[0].split(',')
        lable = [ float(i) for i in lable]
        weights.append(lable)
        
#with open('coef_binary_weights.csv', 'rb') as csvfile:
#    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#    for lable in spamreader:
#        lable = lable[0].split(',')
#        lable = [ float(i) for i in lable]
#        coef.append(lable)

for k in range(0,66):
    sample_tem = copy.deepcopy(sample)
    for m in range(0,2961):
        for n in range(0,199):
            sample_tem[m][n] = sample[m][n]*weights[n][k]#*coef[n][k]
    sample_tem = np.array(sample_tem)
    samples.append(sample_tem)
    sample_tem =[]
    
responses = np.array(responses)

#for z in range (0,66):
#    table = pd.DataFrame(samples[z])
#    table.to_csv('Trainset_weight'+str(z)+'.csv',index = False, header = False)

#samples_trafic = np.array(samples_trafic)
#samples = np.c_[samples,samples_trafic]
#pca = PCA(n_components=100)
#for i in range(0,66):
#    samples[i] = pca.fit(samples[i]).transform(samples[i])

#######################################################################

#######################################################################

#######################################################################
responses_train = responses[1:tt:,]
for x in range(0,66):
    samples_train.append(samples[x][1:tt:,])

responses_test = responses[tt+1:2961:,]
for x in range(0,66):
    samples_test.append(samples[x][tt+1:2961:,])
#########################################################################

#clatrain = samples_train[0:1000:,]
#claresp = responses_train[0:1000:,]
#
#clatest = samples_train[1001:2001:,]
#claresptest = responses_train[1001:2001:,]

KmeanPOI = KmeansPOI()
size = KmeanPOI[0].shape[0]
for r in range (0,6):
    if KmeanPOI[r].shape[0]>=size:
        size = KmeanPOI[r].shape[0]
        size_index = r
        
ind=[2.9,1.0]     
parameter_range = {'C':[0.1,10],'nu':[0.1,1],'tol':[0.001,2],'kernel':['rbf']}
#parameter_range = {'min_samples_split':[1,5],'min_samples_leaf':[1,5]}
svr = svm.SVR()
#svr = RandomForestRegressor(n_estimators=400)
#GRB = GradientBoostingRegressor()
grid = GridSearchCV(svr,parameter_range)
#clf = linear_model.SGDRegressor()
#clf.fit(samples_train,responses_train)
#results = clf.predict(samples_test)
#grid.fit (samples_train[0], responses_train[:,0])


for i in range(0,66):
###    for R in range(0,KmeanPOI[size_index].shape[0]):
#        if i == KmeanPOI[size_index][R]:
    if i not in KmeanPOI[size_index]:
            locals()['clf' + str(i)]= grid
            print 'a'
    else:
        locals()['clf' + str(i)]  = svm.SVR(C=ind[0],kernel='rbf', epsilon = ind[1],gamma =0.001)
#    locals()['clf' + str(i)]  = svm.SVR(C=ind[0],kernel='rbf', epsilon = ind[1],gamma =0.001)    
    locals()['clf' + str(i)].fit (samples_train[i], responses_train[:,i])
##    locals()['clf' + str(i)] = GradientBoostingRegressor(n_estimators=300,learning_rate =2,loss ='huber')
##    locals()['clf' + str(i)] = RandomForestRegressor(n_estimators=200)
##    locals()['clf' + str(i)] = BaggingClassifier(svm.SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False),n_estimators=5,max_samples=0.5, max_features=0.5)    
##    locals()['clf' + str(i)]  = svm.SVR(C=ind[0],kernel='rbf', epsilon = ind[1],gamma =0.001)
##    locals()['clf' + str(i)]  = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)    
##    locals()['clf' + str(i)].fit (samples_train[i], responses_train[:,i])
##    joblib.dump(locals()['clf' + str(i)],'RandomFR_' +str(i)+'.pkl', compress = 3)
    print(i)
##for i in range(0,66):
##    locals()['clf' + str(i)] =joblib.load('RandomFR_' + str(i)+ '.pkl')
#results = grid.predict(samples_test[0])
    
    
results = locals()['clf' + str(0)].predict(samples_test[0])
a = np.abs(responses_test[:,0]-results)
b = responses_test[:,0]-results
sum_ = 0
num_ = 0
for i in range(results.shape[0]):
    if (responses_test[:,0][i] != 0):
        sum_ = sum_ + a[i]/responses_test[:,0][i]
        num_ = num_ + 1
print 'Dist'+str(0)+'eva: %f' %(sum_/num_)


for k in range(1,66):
    locals()['result' + str(k)]= locals()['clf' + str(k)].predict(samples_test[k])
    a = np.abs(responses_test[:,k]-locals()['result' + str(k)]) 
    b = responses_test[:,k]-locals()['result' + str(k)]
    sum_ = 0
    num_ = 0
    for i in range(locals()['result' + str(k)].shape[0]):
        if (responses_test[:,k][i] != 0):
            sum_ = sum_ + a[i]/responses_test[:,k][i]
            num_ = num_ + 1
    print 'Dist'+str(k)+'eva: %f' %(sum_/num_)
    if (sum_/num_)>=0.4:
        over_04_list = np.append(over_04_list,k)
        over_04_value = np.append(over_04_value,(sum_/num_))
    results = np.c_[results,locals()['result' + str(k)]]
    
    
a = np.abs(responses_test-results)
b = responses_test-results
sum_ = 0
num_ = 0
for i in range(results.shape[0]):
    for j in range(results.shape[1]):
        if (responses_test[i,j] != 0):
            sum_ = sum_ + a[i,j]/responses_test[i,j]
            num_ = num_ + 1
print 'eva: %f, ls: %f\n' %(sum_/num_, np.sqrt(np.sum(b*b))/(results.shape[0]*results.shape[1]))

for q in range(0,6):
    for w in KmeanPOI[q]:
        for c in range(over_04_list.shape[0]):
            if w == over_04_list[c]:
                equal = np.append(equal,w)
                equal_value = np.append(equal_value,over_04_value[c])
    equals.append(equal)
    equal_values.append(equal_value)
    print 'Occ'+str(q)+'='+str(equals[q].shape[0])+'+'+str(KmeanPOI[q].shape[0])
    equal = []
    equal_value = []
