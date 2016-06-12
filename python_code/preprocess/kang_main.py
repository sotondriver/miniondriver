#!/usr/bin/env python
# encoding: utf-8
"""
main.py.py

Created by Darcy on 07/06/2016.
Copyright (c) 2016 Darcy. All rights reserved.
"""

import sys
import os
import time
import math
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
from sklearn import datasets, linear_model, metrics
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.learn as skflow

from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DeclarativeBase = declarative_base()


class History(DeclarativeBase):
    """Sqlalchemy deals model"""
    __tablename__ = "history"

    id = Column(Integer, primary_key=True)
    day = Column('day', Integer, nullable=False)
    weekday = Column('weekday', Integer, nullable=False)
    window = Column('window', Integer)
    start_dest = Column('start_dest', Integer)
    end_dest = Column('end_dest', Integer)
    call = Column('call', Integer)
    ans = Column('call', Integer)
    gap = Column('gap', Integer)

engine = create_engine('postgresql://localhost/didi')
# engine.echo = True
DeclarativeBase.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

window22 = [46, 58, 70, 82, 94, 106, 118, 130, 142]
window24 = [58, 70, 82, 94, 106, 118, 130, 142]
window26 = [46, 58, 70, 82, 94, 106, 118, 130, 142]
window28 = [58, 70, 82, 94, 106, 118, 130, 142]
window30 = [46, 58, 70, 82, 94, 106, 118, 130, 142]


def test_data():
    data = []
    for dist in range(1, 67):
        for w in window22:
            d = [22, 5, w, dist]
            data.append(d)

        for w in window24:
            d = [24, 7, w, dist]
            data.append(d)

        for w in window26:
            d = [26, 2, w, dist]
            data.append(d)

        for w in window28:
            d = [28, 4, w, dist]
            data.append(d)

        for w in window30:
            d = [30, 6, w, dist]
            data.append(d)

    test = np.array(data)
    return test


def predict_data(r):
    f = open('predicted.csv', 'w')
    p = 0
    for dist in range(1, 67):
        for w in window22:
            txt = '%d 2016-01-22-%d %f\n' % (dist, w, r[p])
            f.write(txt)
            p += 1

        for w in window24:
            txt = '%d 2016-01-24-%d %f\n' % (dist, w, r[p])
            f.write(txt)
            p += 1

        for w in window26:
            txt = '%d 2016-01-26-%d %f\n' % (dist, w, r[p])
            f.write(txt)
            p += 1

        for w in window28:
            txt = '%d 2016-01-28-%d %f\n' % (dist, w, r[p])
            f.write(txt)
            p += 1

        for w in window30:
            txt = '%d 2016-01-30-%d %f\n' % (dist, w, r[p])
            f.write(txt)
            p += 1
    f.close()


def process_day(cluster_map, day=1, weekday=5):
    day_string= '2016-01-%02.0d' % day
    today = datetime(2016, 1, day, 0, 0)
    print day_string, weekday

    orders = pd.read_table('training_data/order_data/order_data_%s'%day_string, header=None).values
    not_in_dest = 0
    from sets import Set
    not_in_dest_hashs = Set([])

    gaps_call = {}
    gaps_ans = {}

    session = Session()
    for order in orders:
        # print order
        # order = orders[1, :]
        order_id = order[0]
        driver_id = order[1]
        passenger_id = order[2]
        start_district_hash = order[3]
        dest_district_hash = order[4]
        price = order[5]
        t = order[6]

        if not cluster_map.has_key(start_district_hash):
            print 'error start_district_hash', start_district_hash

        start_dest_id = cluster_map[start_district_hash]
        end_dest_id = 0
        if not cluster_map.has_key(dest_district_hash):
            # print 'error dest_district_hash', dest_district_hash
            not_in_dest += 1
            not_in_dest_hashs.add(dest_district_hash)
        else:
            end_dest_id = cluster_map[dest_district_hash]

        date_object = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
        time_to_today = date_object - today
        gap = int(time_to_today.seconds / 600) + 1
        # print type(time_to_today), gap
        # print gap, date_object

        if gaps_call.has_key(gap):
            # gaps_call[gap] += 1
            if gaps_call[gap].has_key(start_dest_id):
                gaps_call[gap][start_dest_id] += 1
            else:
                # print gap, start_dest_id
                gaps_call[gap][start_dest_id] = 1
        else:
            gaps_call[gap] = {start_dest_id: 1}
        # print start_dest_id, type(start_dest_id)
        if len(str(driver_id)) == 32:
            if gaps_ans.has_key(gap):
                if gaps_ans[gap].has_key(start_dest_id):
                    gaps_ans[gap][start_dest_id] += 1
                else:
                    gaps_ans[gap][start_dest_id] = 1
            else:
                gaps_ans[gap] = {start_dest_id: 1}
    # print gaps_call
    # print gaps_ans
    data = []
    for i in range(1, 145):
        # print key
        # print gaps_call[i], gaps_ans[i]
        for j in range(1, 67):
            c = 0
            a = 0
            if not gaps_call[i].has_key(j):
                pass
            else:
                c = gaps_call[i][j]
            if not gaps_ans[i].has_key(j):
                pass
            else:
                a = gaps_ans[i][j]

            if c < a:
                print i, j, c, a
            else:
                # print i, j, c - a
                d = [i, j, c - a]
                data.append(d)
                h = History()
                h.day = day
                h.weekday = weekday
                h.window = i
                h.start_dest = j
                h.call = c
                h.ans = a
                h.gap = c - a
                session.add(h)

    # dataset = np.array(data)
    # x = dataset[:, 0:2]
    # y = dataset[:, 2]
    # print x
    # print y

    session.commit()
    session.close()

    print 'total:', len(data)

    print not_in_dest, len(orders), len(not_in_dest_hashs)
    print


def import_data():
    cluster_map_values = pd.read_table('training_data/cluster_map/cluster_map', header=None).values
    # print cluster_map_values
    cluster_map = {}
    for v in cluster_map_values:
        # print v[0], v[1]
        cluster_map[v[0]] = v[1]
    print cluster_map

    for day in range(1, 22):
        weekday = (day+3)%7 + 1
        process_day(cluster_map, day=day, weekday=weekday)


def build_data():
    session = Session()
    histories = session.query(History).all()
    data = []
    for h in histories:
        # print h
        d = [h.day, h.weekday, h.window, h.start_dest, h.gap]
        data.append(d)
    dataset = np.array(data)
    np.savetxt('dataset.csv', dataset, delimiter=', ', fmt ='%d')
    session.close()


def train():
    dataset = pd.read_csv('dataset.csv', header=None).values
    x = dataset[5:1*144*66, 0:4]
    y = dataset[5:1*144*66, 4]

    tx = dataset[-2*144*66:, 0:4]
    ty = dataset[-2*144*66:, 4]
    print len(ty)
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    print('Coefficients: \n', regr.coef_)

    # predicted = regr.predict(tx)
    print("Residual sum of squares: %.2f" % np.mean((regr.predict(tx) - ty) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(tx, ty))
    #test = test_data()
    #r = regr.predict(test)
    # predict_data(r)
    # x = np.arange(len(y))
    # plt.plot(x, y, color='blue')
    # plt.show()


def tf_train():
    dataset = pd.read_csv('dataset.csv', header=None).values.astype(np.float)
    train_X = dataset[:-2*144*66, 0:4]
    train_Y = dataset[:-2*144*66, 4]

    tx = dataset[-2*144*66:, 0:4]
    ty = dataset[-2*144*66:, 4]

    print train_Y
    n_samples = len(train_Y)
    print dataset.shape
    regressor = skflow.TensorFlowLinearRegressor()
    regressor.fit(train_X, train_Y)
    score = metrics.mean_squared_error(regressor.predict(tx), ty)
    print ("MSE: %f" % score)


def main():
    start = time.time()

    # import_data()
    build_data()
    train()
    # tf_train()

    # print y
    # print x
    # plt.plot(x, y, color='blue', linewidth=3)
    # plt.xticks(())
    # plt.yticks(())
    # plt.show()

    end = time.time()
    print 'CPU time: %f seconds.' % (end - start,)


if '__main__' == __name__:
    main()
