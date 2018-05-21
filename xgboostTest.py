#!/usr/bin/python
#-*- coding:utf-8 -*-

import  pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

import scipy as sp


# 训练分类器XGB
def trainClassifierXGB(x_train, y_train):
    print('使用XGBOOST进行训练')

    dtrain = xgb.DMatrix(x_train, label=y_train)

    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'logloss',
              'gamma': 0.08,
              'min_child_weight': 1.1,
              'max_depth': 4,
              'lambda': 10,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'colsample_bylevel': 0.7,
              'eta': 0.01,
              'tree_method': 'exact',
              'seed': 0,
              'nthread': 12
              }

    plst = list(params.items())
    plst += [('eval_metric', 'auc')]  # Multiple evals can be handled in this way
    plst += [('eval_metric', 'ams@0')]

    num_round = 1200
    bst = xgb.train(plst, dtrain, num_boost_round=num_round)

    print(params)
    print(num_round)

    return bst


def predict_test_prob(bst):
    df_all_test = pd.read_csv('data_csv/final3_encode.csv')
    df_all_test.fillna(value=0, inplace=True)
    df_all_test.drop_duplicates(inplace=True)

    df_sta_lgbm = pd.read_csv('C:/Users/user/Desktop/ijcai_result/prob_lgbm_test.txt',sep=' ')

    print('开始拼接')
    df_all = pd.merge(df_all_test, df_sta_lgbm, how='left', on='instance_id')

    del df_sta_lgbm
    instanceID = df_all.instance_id.values
    feature_all = df_all.drop(['instance_id','user_id','time','is_trade',
                               'context_timestamp'], axis=1).values

    del df_all

    dtest=xgb.DMatrix(feature_all)

    prob = bst.predict(dtest)

    output = pd.DataFrame({'instance_id': instanceID, 'predicted_score': prob})

    output.to_csv('C:/Users/user/Desktop/ijcai_result/result_xgb_test.txt',sep=' ',index=False)


# 交叉验证
def cross_validat(df_all, test_size=0.3):


    feature_all = df_all.drop(['instance_id','user_id','time','is_trade',
                               'context_timestamp'], axis=1).values

    label_all = df_all.is_trade.values

    del df_all

    x_train, x_test, y_train, y_test = train_test_split(feature_all, label_all, test_size=test_size, random_state=0)
    print('数据集切割完成')

    del feature_all
    del label_all

    bst = trainClassifierXGB(x_train, y_train)
    print('训练完成')
    print(bst.get_score())

    dtest = xgb.DMatrix(x_test)

    prob = bst.predict(dtest)

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

    df_sta_lgbm = pd.read_csv('C:/Users/user/Desktop/ijcai_result/prob_lgbm_train.txt',sep=' ')
    print('开始拼接')

    df_all = pd.merge(df_all, df_sta_lgbm, how='left', on='instance_id')
    del df_sta_lgbm

    print('开始训练')

    pre=cross_validat(df_all,test_size=0.3)

    feature_all = df_all.drop(['instance_id','user_id','time','is_trade',
                               'context_timestamp'], axis=1).values
    label_all = df_all.is_trade.values

    del df_all

    bst = trainClassifierXGB(feature_all, label_all)
    print('分类器训练完成')
    predict_test_prob(bst)
    print('结果预测完成')
