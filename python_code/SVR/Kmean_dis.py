# -*- coding: utf-8 -*-
"""
Created on Wed Jun 08 23:06:18 2016

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
from sklearn import svm
from sklearn.cluster import KMeans
from collections import Counter
import pylab as pl
import pandas as pd


def KmeansPOI():
    poi = []
    clf = []
    LI =[[],[],[],[],[],[]]
    
    with open('poi_data.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            row = row[0].split(',')
            row = [int(i) for i in row ]    
            poi.append(row)
    clf = KMeans(n_clusters = 6,max_iter = 4000,n_init = 10,verbose = 1)
    poi = np.array(poi)
    a = clf.fit_predict(poi)
    A = Counter(a)
    b = A.most_common(1)
    c = [i for i in range (1,12)]
    
    for i in range(0,66):
        LI[a[i]] = np.append(LI[a[i]],i)
    return(LI)



