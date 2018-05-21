#!/usr/bin/python
#-*- coding:utf-8 -*-

import  pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

import time

pd.set_option('display.width',1000)


def timestamp_datetime(value):
    format='%Y-%m-%d %H:%M:%S'
    value=time.localtime(value)
    dt=time.strftime(format,value)
    return dt


def convert_data(data):
    data['time'] = data.context_timestamp.apply(timestamp_datetime)
    data['day'] = data.time.apply(lambda x: int(x[8:10]))
    data['hour'] = data.time.apply(lambda x: int(x[11:13]))
    data['minute'] = data.time.apply(lambda x: int(x[14:16]))
    return data

def heat_map(data):

    train_data = data.drop(['instance_id', 'context_timestamp', 'time'], axis=1)
    corrmat = train_data.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    ax.set_xticklabels(corrmat, rotation='horizontal')
    sns.heatmap(corrmat, vmax=.8, square=True)
    label_y = ax.get_yticklabels()
    plt.setp(label_y, rotation=360)
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=90)
    plt.show()

if __name__=="__main__":

    train_data = pd.read_table('data_csv/round1_ijcai_18_train_20180301.txt', sep=' ')
    train_data = convert_data(train_data)
    train_data.drop_duplicates(inplace=True)





    # test_a = pd.read_csv('data_csv/round1_ijcai_18_test_a_20180301.txt', sep=' ')
    # test_b = pd.read_csv('data_csv/round1_ijcai_18_test_b_20180418.txt', sep=' ')
    # test = pd.concat([test_a, test_b], axis=0)
    # test_data = convert_data(test)

    # heat_map(train_data)










