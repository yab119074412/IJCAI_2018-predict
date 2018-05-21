#!/usr/bin/python
#-*- coding:utf-8 -*-

import  pandas as pd
import numpy as np
import time
import seaborn as sns
import random
from sklearn.metrics import log_loss
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

sns.set(style='ticks')
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


def add_time_diff(group):
    m = len(group)
    group = group.sort_values(by='context_timestamp',ascending=True)
    return group.iloc[0:m-1,:]

# df_total_data = df_total_data.groupby('user_id').apply(add_time_diff)


def user_feature(data):

    t = data[['user_id','item_id','shop_id','day','hour']]
    t['same_user_click_times'] = 1
    t=t.groupby(['user_id','item_id','shop_id','day','hour']).agg('sum').reset_index()

    k=t[t['same_user_click_times'] >1]
    k = pd.merge(k,data,on=['user_id','item_id','shop_id','day','hour'],how='left')
    feature = ['instance_id','user_id','item_id','shop_id','time','context_timestamp','day','hour','minute']

    da = k[feature]

    da = da.groupby(['user_id','item_id','shop_id']).apply(add_time_diff)

    print(da)
    # da.to_csv('C:/Users/user/Desktop/same_minute_click.csv', index=False)

    return da
    # da.to_csv('C:/Users/user/Desktop/same_minute_click.csv',index=False)


def score(x,list):
    if x in list:
        x=x/100
    return x


if __name__=="__main__":

    train_data = pd.read_table('data_csv/round1_ijcai_18_train_20180301.txt', sep=' ')
    train_data = convert_data(train_data)
    train_data.drop_duplicates(inplace=True)

    # dataset2 = train_data.loc[(train_data.day == 23) | (train_data.day == 24)]
    # da = user_feature(dataset2)

    dataset3 = pd.read_csv('data_csv/round1_ijcai_18_test_b_20180418.txt', sep=' ')
    dataset3 = convert_data(dataset3)
    da = user_feature(dataset3)

    result = pd.read_csv('C:/Users/user/Desktop/submission_xgb1500.4.22.txt',sep=' ')
    print(result)

    # result = pd.read_csv('C:/Users/user/Desktop/ijcai_result/result_20180423.txt', sep=' ')

    # dataset = dataset3[['instance_id','is_trade']]

    # new_result = pd.merge(result,dataset,on='instance_id',how='left')

    # print(log_loss(new_result['is_trade'], new_result['predicted_score']))

    # print(log_loss(new_result['is_trade'], new_result['predicted_score']))

    instance_id = list(da['instance_id'])

    for i in range(result.shape[0]):
        if result['instance_id'][i] in instance_id:
    #     if result['instance_id'][i] >0.05:
    #         result['predicted_score'][i] = result['predicted_score'][i] * 10
    #     if result['instance_id'][i] <0.01:
            result['predicted_score'][i] = result['predicted_score'][i] / 5

    # result['predicted_score'] = result['predicted_score'].apply(lambda x:score(x,instance_id))
    # print(log_loss(result['is_trade'],result['predicted_score']))


    # for i in range(result.shape[0]):
    #     result['predicted_score'][i] = ("%.9f" % result['predicted_score'][i])

    # submit = result.iloc[18371:, :]
    result.to_csv('C:/Users/user/Desktop/submission_xgb1500.4.22_1.txt', sep=' ', index=False)
    # print(submit)