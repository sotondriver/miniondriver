import matplotlib.pyplot as plt  
import numpy as np  
from sklearn import linear_model  

from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation  
from keras.optimizers import SGD  
  
population = [54167,55196,56300,57482,58796,60266,61465,62828,64653,65994,67207,66207,65859,67295,69172,70499,72538,74542,76368,78534,80671,82992,85229,87177,89211,90859,92420,93717,94974,96259,97542,98705,100072,101654,103008,104357,105851,107507,109300,111026,112704,114333,115823,117171,118517,119850,121121,122389,123626,124761,125786,126743,127627,128453,129227,129988,130756,131448,132129,132802,134480,135030,135770,136460,137510]  
lag = 3
N = len(population)-lag
train_data = np.zeros((N,lag), dtype=np.int)
train_label = np.zeros((N,1), dtype=np.int)
for i in range(N):
    train_data[i,:] = population[i:i+lag]
    train_label[i] = population[i+lag]

clf = linear_model.LinearRegression()   
clf.fit(train_data, train_label)
predict_ = clf.predict(train_data)
print 'eva: %f' %(np.sum(np.abs(predict_-train_label))/len(train_label))


model = Sequential()
model.add(Dense(20, input_dim=lag,activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(train_data, train_label, nb_epoch=100, batch_size=16)
predict_ = model.predict(train_data)
print 'eva: %f' %(np.sum(np.abs(predict_-train_label))/len(train_label))

