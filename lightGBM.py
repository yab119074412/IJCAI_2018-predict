#!/usr/bin/python
#-*- coding:utf-8 -*-

import  pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.utils import shuffle
from sklearn.metrics import log_loss

pd.set_option('display.width',1000)


# 训练分类器XGB
def trainClassifierLGBM(x_train, y_train):

    print('使用lightgbm进行训练')

    lgb_train = lgb.Dataset(x_train,y_train)

    params={
        'task':'train',
        'boosting_type':'gbdt',
        'objective':'binary',
        'metric':'binary_logloss',
        'num_leaves':10,
        'max_depth':4,
        'learning_rate':0.01,
        'n_estimators':1000,
        'colsample_bytree':1.0,
    }

    lgbm = lgb.train(params,lgb_train,num_boost_round=700)

    print(params)
    return lgbm

def cutData():
    dataset1 = pd.read_csv('data_csv/final1_encode.csv')
    dataset2 = pd.read_csv('data_csv/final2_encode.csv')

    dataset1.fillna(value=0, inplace=True)
    dataset2.fillna(value=0, inplace=True)

    dataset1.drop_duplicates(inplace=True)
    dataset2.drop_duplicates(inplace=True)

    print('开始拼接')
    dataset12 = pd.concat([dataset1, dataset2], axis=0)

    del dataset1, dataset2

    dataset12 = shuffle(dataset12, random_state=0)
    step = len(dataset12) // 10

    train1 = dataset12[0:step]
    train2 = dataset12[step:2 * step]
    train3 = dataset12[2 * step:3 * step]
    train4 = dataset12[3 * step:4 * step]
    train5 = dataset12[4 * step:5 * step]
    train6 = dataset12[5 * step:6 * step]
    train7 = dataset12[6 * step:7 * step]
    train8 = dataset12[7 * step:8 * step]
    train9 = dataset12[8 * step:9 * step]
    train10 = dataset12[9 * step:]

    del dataset12
    return train1, train2, train3, train4, train5, train6, train7, train8, train9, train10


def stacking_test():
    dataset1 = pd.read_csv('data_csv/final1_encode.csv')
    dataset2 = pd.read_csv('data_csv/final2_encode.csv')

    dataset1.fillna(value=0, inplace=True)
    dataset2.fillna(value=0, inplace=True)

    dataset1.drop_duplicates(inplace=True)
    dataset2.drop_duplicates(inplace=True)

    print('开始拼接')
    df_all = pd.concat([dataset1, dataset2], axis=0)
    del dataset1, dataset2

    feature_all = df_all.drop(['instance_id', 'user_id', 'time', 'is_trade',
                               'context_timestamp'], axis=1)

    label_all = df_all['is_trade']

    del df_all

    lgbm = trainClassifierLGBM(feature_all, label_all)

    df_all_test = pd.read_csv('data_csv/final3_encode.csv')
    df_all_test.fillna(value=0, inplace=True)
    df_all_test.drop_duplicates(inplace=True)

    feature_all = df_all_test.drop(['instance_id', 'user_id', 'time', 'context_timestamp', 'is_trade'], axis=1)

    instanceID = df_all_test.instance_id.values
    prob = lgbm.predict(feature_all,num_iteration=lgbm.best_iteration)

    df_prob = pd.DataFrame({'instance_id': instanceID, 'predicted_score': prob})
    df_prob.to_csv('C:/Users/user/Desktop/ijcai_result/prob_lgbm_test.txt',sep=' ',index=False)


def stacking_train():
    train1, train2, train3, train4, train5, train6, train7, train8, train9, train10 = cutData()

    print('分割完成')

    list_instance = np.concatenate((train1['instance_id'].values, train2['instance_id'].values,
                                    train3['instance_id'].values, train4['instance_id'].values,
                                    train5['instance_id'].values, train6['instance_id'].values,
                                    train7['instance_id'].values, train8['instance_id'].values,
                                    train9['instance_id'].values, train10['instance_id'].values))

    list_prob = []
    # 训练
    print('开始stacking1')
    strain = train2.append([train3, train4, train5,train6,train7,train8,train9,train10])
    stest = train1
    label_all = strain['is_trade'].values
    feature_all = strain.drop(['instance_id','user_id','time','is_trade',
                               'context_timestamp'], axis=1).values
    del strain
    lgbm = trainClassifierLGBM(feature_all, label_all)

    del feature_all
    del label_all

    test_all = stest.drop(['instance_id','user_id','time','is_trade',
                               'context_timestamp'], axis=1).values


    prob = lgbm.predict(test_all,num_iteration=lgbm.best_iteration)
    print(log_loss(stest['is_trade'].values,prob))

    del stest
    list_prob += prob.tolist()

    print(len(list_prob))
    del test_all
    del prob
    del lgbm

    print('开始stacking2')
    strain = train1.append([train3, train4, train5,train6,train7,train8,train9,train10])
    stest = train2
    label_all = strain['is_trade'].values
    feature_all = strain.drop(['instance_id','user_id','time','is_trade',
                               'context_timestamp'], axis=1).values
    del strain
    lgbm = trainClassifierLGBM(feature_all, label_all)
    del feature_all
    del label_all
    test_all = stest.drop(['instance_id','user_id','time','is_trade',
                               'context_timestamp'], axis=1).values


    prob = lgbm.predict(test_all,num_iteration=lgbm.best_iteration)
    print(log_loss(stest['is_trade'].values, prob))

    del stest
    list_prob += prob.tolist()

    print(len(list_prob))
    del test_all
    del prob
    del lgbm

    print('开始stacking3')
    strain = train1.append([train2, train4, train5,train6,train7,train8,train9,train10])
    stest = train3
    label_all = strain['is_trade'].values
    feature_all = strain.drop(['instance_id','user_id','time','is_trade',
                               'context_timestamp'], axis=1).values
    del strain
    lgbm = trainClassifierLGBM(feature_all, label_all)
    del feature_all
    del label_all
    test_all = stest.drop(['instance_id','user_id','time','is_trade',
                               'context_timestamp'], axis=1).values

    prob = lgbm.predict(test_all,num_iteration=lgbm.best_iteration)
    print(log_loss(stest['is_trade'].values, prob))
    del stest
    list_prob += prob.tolist()

    print(len(list_prob))
    del test_all
    del prob
    del lgbm

    print('开始stacking4')
    strain = train1.append([train2, train3, train5,train6,train7,train8,train9,train10])
    stest = train4
    label_all = strain['is_trade'].values
    feature_all = strain.drop(['instance_id','user_id','time','is_trade',
                               'context_timestamp'], axis=1).values
    del strain
    lgbm = trainClassifierLGBM(feature_all, label_all)
    del feature_all
    del label_all
    test_all = stest.drop(['instance_id','user_id','time','is_trade',
                               'context_timestamp'], axis=1).values

    prob = lgbm.predict(test_all,num_iteration=lgbm.best_iteration)
    print(log_loss(stest['is_trade'].values, prob))
    del stest
    list_prob += prob.tolist()

    print(len(list_prob))
    del test_all
    del prob
    del lgbm

    print('开始stacking5')
    strain = train1.append([train2, train3, train4,train6,train7,train8,train9,train10])
    stest = train5
    label_all = strain['is_trade'].values
    feature_all = strain.drop(['instance_id','user_id','time','is_trade',
                               'context_timestamp'], axis=1).values
    del strain
    lgbm = trainClassifierLGBM(feature_all, label_all)
    del feature_all
    del label_all
    test_all = stest.drop(['instance_id','user_id','time','is_trade',
                               'context_timestamp'], axis=1).values

    prob = lgbm.predict(test_all)
    print(log_loss(stest['is_trade'].values, prob))
    del stest
    list_prob += prob.tolist()

    print(len(list_prob))
    del test_all
    del prob
    del lgbm

    print('开始stacking6')
    strain = train1.append([train2, train3, train4, train5, train7, train8, train9, train10])
    stest = train6
    label_all = strain['is_trade'].values
    feature_all = strain.drop(['instance_id', 'user_id', 'time', 'is_trade',
                               'context_timestamp'], axis=1).values
    del strain
    lgbm = trainClassifierLGBM(feature_all, label_all)
    del feature_all
    del label_all
    test_all = stest.drop(['instance_id', 'user_id', 'time', 'is_trade',
                           'context_timestamp'], axis=1).values

    prob = lgbm.predict(test_all)
    print(log_loss(stest['is_trade'].values, prob))
    del stest
    list_prob += prob.tolist()

    print(len(list_prob))
    del test_all
    del prob
    del lgbm

    print('开始stacking7')
    strain = train1.append([train2, train3, train4, train5, train6, train8, train9, train10])
    stest = train7
    label_all = strain['is_trade'].values
    feature_all = strain.drop(['instance_id', 'user_id', 'time', 'is_trade',
                               'context_timestamp'], axis=1).values
    del strain
    lgbm = trainClassifierLGBM(feature_all, label_all)
    del feature_all
    del label_all
    test_all = stest.drop(['instance_id', 'user_id', 'time', 'is_trade',
                           'context_timestamp'], axis=1).values

    prob = lgbm.predict(test_all)
    print(log_loss(stest['is_trade'].values, prob))
    del stest
    list_prob += prob.tolist()

    print(len(list_prob))
    del test_all
    del prob
    del lgbm

    print('开始stacking8')
    strain = train1.append([train2, train3, train4, train5, train6, train7, train9, train10])
    stest = train8
    label_all = strain['is_trade'].values
    feature_all = strain.drop(['instance_id', 'user_id', 'time', 'is_trade',
                               'context_timestamp'], axis=1).values
    del strain
    lgbm = trainClassifierLGBM(feature_all, label_all)
    del feature_all
    del label_all
    test_all = stest.drop(['instance_id', 'user_id', 'time', 'is_trade',
                           'context_timestamp'], axis=1).values

    prob = lgbm.predict(test_all)
    print(log_loss(stest['is_trade'].values, prob))
    del stest
    list_prob += prob.tolist()

    print(len(list_prob))
    del test_all
    del prob
    del lgbm

    print('开始stacking9')
    strain = train1.append([train2, train3, train4, train5, train6, train7, train8, train10])
    stest = train9
    label_all = strain['is_trade'].values
    feature_all = strain.drop(['instance_id', 'user_id', 'time', 'is_trade',
                               'context_timestamp'], axis=1).values
    del strain
    lgbm = trainClassifierLGBM(feature_all, label_all)
    del feature_all
    del label_all
    test_all = stest.drop(['instance_id', 'user_id', 'time', 'is_trade',
                           'context_timestamp'], axis=1).values

    prob = lgbm.predict(test_all)
    print(log_loss(stest['is_trade'].values, prob))
    del stest
    list_prob += prob.tolist()

    print(len(list_prob))
    del test_all
    del prob
    del lgbm

    print('开始stacking10')
    strain = train1.append([train2, train3, train4, train5, train6, train7, train8, train9])
    stest = train10
    label_all = strain['is_trade'].values
    feature_all = strain.drop(['instance_id', 'user_id', 'time', 'is_trade',
                               'context_timestamp'], axis=1).values
    del strain
    lgbm = trainClassifierLGBM(feature_all, label_all)
    del feature_all
    del label_all
    test_all = stest.drop(['instance_id', 'user_id', 'time', 'is_trade',
                           'context_timestamp'], axis=1).values

    prob = lgbm.predict(test_all)
    print(log_loss(stest['is_trade'].values, prob))
    del stest
    list_prob += prob.tolist()

    print(len(list_prob))
    del test_all
    del prob
    del lgbm


    del train1
    del train2
    del train3
    del train4
    del train5
    del train6
    del train7
    del train8
    del train9
    del train10

    df_prob = pd.DataFrame({'instance_id': list_instance, 'predicted_score': list_prob})

    df_prob.to_csv('C:/Users/user/Desktop/ijcai_result/prob_lgbm_train.txt',sep=' ',index=False)


if __name__=="__main__":

    # stacking_train()
    stacking_test()



