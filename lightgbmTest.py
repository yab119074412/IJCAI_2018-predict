#!/usr/bin/python
#-*- coding:utf-8 -*-

import  pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from ijcai_search.TiaoCan import RPTune
from sklearn.grid_search import RandomizedSearchCV

import scipy as sp


# 训练分类器XGB
def trainClassifierLGBM(x_train, y_train):
    print('使用LIGHTBGM进行训练')

    lgb_train = lgb.Dataset(x_train, y_train)

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 10,
        'max_depth': 4,
        'learning_rate': 0.01,
        'colsample_bytree': 1.0,
    }

    lgbm = lgb.train(params,
                     lgb_train,
                     num_boost_round=700)

    print(params)

    return lgbm


def predict_test_prob(lgbm):
    df_all_test = pd.read_csv('data_csv/final3_encode.csv')
    df_all_test.fillna(value=0, inplace=True)
    df_all_test.drop_duplicates(inplace=True)

    df_sta_xgb = pd.read_csv('C:/Users/user/Desktop/ijcai_result/prob_xgb_test.txt',sep=' ')

    print('开始拼接')
    df_all = pd.merge(df_all_test, df_sta_xgb, how='left', on='instance_id')

    del df_sta_xgb
    instanceID = df_all.instance_id.values
    feature_all = df_all.drop(['instance_id','user_id','time','context_timestamp','is_trade'], axis=1).values

    prob = lgbm.predict(feature_all, num_iteration=lgbm.best_iteration)

    output = pd.DataFrame({'instance_id': instanceID, 'predicted_score': prob})

    output.to_csv("C:/Users/user/Desktop/ijcai_result/result_lgbm_test.txt", sep=' ', index=False)



# 交叉验证
def cross_validat(df_all, test_size=0.3):

    feature_all = df_all.drop(['instance_id','user_id','time','is_trade','context_timestamp'], axis=1).values

    label_all = df_all.is_trade.values

    del df_all

    x_train, x_test, y_train, y_test = train_test_split(feature_all, label_all, test_size=test_size, random_state=1)
    print('数据集切割完成')

    del feature_all
    del label_all

    lgbm = trainClassifierLGBM(x_train, y_train)
    print('训练完成')

    prob = lgbm.predict(x_test, num_iteration=lgbm.best_iteration)

    loss = log_loss(y_test, prob)
    print('交叉验证损失为:', loss)

    return prob


# 主函数入口
if __name__ == '__main__':
    dataset1 = pd.read_csv('data_csv/final1_encode.csv')
    dataset2 = pd.read_csv('data_csv/final2_encode.csv')

    dataset1.fillna(value=0, inplace=True)
    dataset2.fillna(value=0, inplace=True)

    dataset1.drop_duplicates(inplace=True)
    dataset2.drop_duplicates(inplace=True)

    df_all = pd.concat([dataset1, dataset2], axis=0)
    del dataset1, dataset2

    df_sta_xgb = pd.read_csv('C:/Users/user/Desktop/ijcai_result/prob_xgb_train.txt',sep=' ')
    print('开始拼接')

    df_all = pd.merge(df_all, df_sta_xgb, how='left', on='instance_id')
    del df_sta_xgb

    print('开始训练')

    pre=cross_validat(df_all,test_size=0.3)


    feature_all = df_all.drop(['instance_id','user_id','time','is_trade','context_timestamp'], axis=1).values
    label_all = df_all.is_trade.values

    del df_all

    bst = trainClassifierLGBM(feature_all, label_all)
    print('分类器训练完成')
    predict_test_prob(bst)
    print('结果预测完成')
