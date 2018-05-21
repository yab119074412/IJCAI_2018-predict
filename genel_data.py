#!/usr/bin/python
#-*- coding:utf-8 -*-

import  pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.utils import shuffle
from sklearn.metrics import log_loss

pd.set_option('display.width',1000)


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
              'scale_pos_weight': 1,
              'nthread': 12
              }

    # 不用设置线程，XGboost会自行设置所有的
    # param['nthread'] = 8

    plst = list(params.items())
    plst += [('eval_metric', 'auc')]  # Multiple evals can be handled in this way
    plst += [('eval_metric', 'ams@0')]

    num_round = 1200
    bst = xgb.train(plst, dtrain, num_boost_round=num_round)

    print(params)
    print(num_round)

    return bst

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
    return train1,train2,train3,train4,train5,train6,train7,train8,train9,train10


def stacking_test():
    dataset1 = pd.read_csv('data_csv/final1_encode.csv')
    dataset2 = pd.read_csv('data_csv/final2_encode.csv')

    dataset1.fillna(value=0, inplace=True)
    dataset2.fillna(value=0, inplace=True)

    dataset1.drop_duplicates(inplace=True)
    dataset2.drop_duplicates(inplace=True)

    print('开始拼接')
    df_all = pd.concat([dataset1,dataset2],axis=0)
    del dataset1,dataset2

    feature_all = df_all.drop(['instance_id','user_id','time','is_trade',
                               'context_timestamp'], axis=1)

    label_all = df_all['is_trade']

    del df_all

    lgbm = trainClassifierXGB(feature_all, label_all)

    df_all_test = pd.read_csv('data_csv/final3_encode.csv')
    df_all_test.fillna(value=0, inplace=True)
    df_all_test.drop_duplicates(inplace=True)

    feature_all = df_all_test.drop(['instance_id','user_id','time','context_timestamp','is_trade'], axis=1)

    instanceID = df_all_test.instance_id.values
    dtest = xgb.DMatrix(feature_all)

    del df_all_test

    prob = lgbm.predict(dtest)

    df_prob = pd.DataFrame({'instance_id': instanceID, 'predicted_score': prob})
    df_prob.to_csv('C:/Users/user/Desktop/ijcai_result/prob_xgb_test.txt',sep=' ',index=False)


def stacking_train():
    train1, train2, train3, train4, train5, train6,train7,train8,train9,train10 = cutData()

    print('分割完成')

    list_instance = np.concatenate((train1['instance_id'].values, train2['instance_id'].values,
                                    train3['instance_id'].values, train4['instance_id'].values,
                                    train5['instance_id'].values,train6['instance_id'].values,
                                    train7['instance_id'].values,train8['instance_id'].values,
                                    train9['instance_id'].values,train10['instance_id'].values))

    list_prob = []
    # 训练
    print('开始stacking1')
    strain = train2.append([train3, train4, train5,train6,train7,train8,train9,train10])
    stest = train1
    label_all = strain['is_trade'].values
    feature_all = strain.drop(['instance_id','user_id','time','is_trade',
                               'context_timestamp'], axis=1).values
    del strain
    lgbm = trainClassifierXGB(feature_all, label_all)

    del feature_all
    del label_all

    test_all = stest.drop(['instance_id','user_id','time','is_trade','context_timestamp'], axis=1).values
    dtest = xgb.DMatrix(test_all)
    del test_all
    prob = lgbm.predict(dtest)
    loss1 = log_loss(stest['is_trade'].values,prob)
    print(loss1)
    del stest
    list_prob += prob.tolist()

    print(len(list_prob))
    del dtest
    del prob
    del lgbm

    print('开始stacking2')
    strain = train1.append([train3, train4, train5,train6,train7,train8,train9,train10])
    stest = train2
    label_all = strain['is_trade'].values
    feature_all = strain.drop(['instance_id','user_id','time','is_trade','context_timestamp'], axis=1).values
    del strain
    lgbm = trainClassifierXGB(feature_all, label_all)
    del feature_all
    del label_all
    test_all = stest.drop(['instance_id','user_id','time','is_trade',
                               'context_timestamp'], axis=1).values

    dtest = xgb.DMatrix(test_all)
    del test_all
    prob = lgbm.predict(dtest)
    loss2 = log_loss(stest['is_trade'].values, prob)
    print(loss2)
    del stest
    list_prob += prob.tolist()

    print(len(list_prob))
    del dtest
    del prob
    del lgbm

    print('开始stacking3')
    strain = train1.append([train2, train4, train5,train6,train7,train8,train9,train10])
    stest = train3
    label_all = strain['is_trade'].values
    feature_all = strain.drop(['instance_id','user_id','time','is_trade',
                               'context_timestamp'], axis=1).values
    del strain
    lgbm = trainClassifierXGB(feature_all, label_all)
    del feature_all
    del label_all
    test_all = stest.drop(['instance_id','user_id','time','is_trade',
                               'context_timestamp'], axis=1).values
    dtest = xgb.DMatrix(test_all)
    del test_all
    prob = lgbm.predict(dtest)
    loss3 = log_loss(stest['is_trade'].values, prob)
    print(loss3)
    del stest
    list_prob += prob.tolist()

    print(len(list_prob))
    del dtest
    del prob
    del lgbm

    print('开始stacking4')
    strain = train1.append([train2, train3, train5,train6,train7,train8,train9,train10])
    stest = train4
    label_all = strain['is_trade'].values
    feature_all = strain.drop(['instance_id','user_id','time','is_trade',
                               'context_timestamp'], axis=1).values
    del strain
    lgbm = trainClassifierXGB(feature_all, label_all)
    del feature_all
    del label_all
    test_all = stest.drop(['instance_id','user_id','time','is_trade',
                               'context_timestamp'], axis=1).values
    dtest = xgb.DMatrix(test_all)
    del test_all
    prob = lgbm.predict(dtest)
    loss4=log_loss(stest['is_trade'].values, prob)
    print(loss4)
    del stest
    list_prob += prob.tolist()

    print(len(list_prob))
    del dtest
    del prob
    del lgbm

    print('开始stacking5')
    strain = train1.append([train2, train3, train4,train6,train7,train8,train9,train10])
    stest = train5
    label_all = strain['is_trade'].values
    feature_all = strain.drop(['instance_id','user_id','time','is_trade',
                               'context_timestamp'], axis=1).values
    del strain
    lgbm = trainClassifierXGB(feature_all, label_all)
    del feature_all
    del label_all
    test_all = stest.drop(['instance_id','user_id','is_trade','time',
                               'context_timestamp'], axis=1).values
    dtest = xgb.DMatrix(test_all)
    del test_all
    prob = lgbm.predict(dtest)
    loss5 = log_loss(stest['is_trade'].values, prob)
    print(loss5)

    del stest
    list_prob += prob.tolist()

    print(len(list_prob))
    del dtest
    del prob
    del lgbm

    print('开始stacking6')
    strain = train1.append([train2, train3, train4, train5, train7, train8, train9, train10])
    stest = train6
    label_all = strain['is_trade'].values
    feature_all = strain.drop(['instance_id', 'user_id', 'time', 'is_trade',
                               'context_timestamp'], axis=1).values
    del strain
    lgbm = trainClassifierXGB(feature_all, label_all)
    del feature_all
    del label_all
    test_all = stest.drop(['instance_id', 'user_id', 'time', 'is_trade',
                           'context_timestamp'], axis=1).values
    dtest = xgb.DMatrix(test_all)
    del test_all
    prob = lgbm.predict(dtest)
    loss6 = log_loss(stest['is_trade'].values, prob)
    print(loss6)
    del stest
    list_prob += prob.tolist()

    print(len(list_prob))
    del dtest
    del prob
    del lgbm

    print('开始stacking7')
    strain = train1.append([train2, train3, train4, train5, train6, train8, train9, train10])
    stest = train7
    label_all = strain['is_trade'].values
    feature_all = strain.drop(['instance_id', 'user_id', 'time', 'is_trade',
                               'context_timestamp'], axis=1).values
    del strain
    lgbm = trainClassifierXGB(feature_all, label_all)
    del feature_all
    del label_all
    test_all = stest.drop(['instance_id', 'user_id', 'time', 'is_trade',
                           'context_timestamp'], axis=1).values
    dtest = xgb.DMatrix(test_all)
    del test_all
    prob = lgbm.predict(dtest)
    loss7 = log_loss(stest['is_trade'].values, prob)
    print(loss7)
    del stest
    list_prob += prob.tolist()

    print(len(list_prob))
    del dtest
    del prob
    del lgbm

    print('开始stacking8')
    strain = train1.append([train2, train3, train4, train5, train6, train7, train9, train10])
    stest = train8
    label_all = strain['is_trade'].values
    feature_all = strain.drop(['instance_id', 'user_id', 'time', 'is_trade',
                               'context_timestamp'], axis=1).values
    del strain
    lgbm = trainClassifierXGB(feature_all, label_all)
    del feature_all
    del label_all
    test_all = stest.drop(['instance_id', 'user_id', 'time', 'is_trade',
                           'context_timestamp'], axis=1).values
    dtest = xgb.DMatrix(test_all)
    del test_all
    prob = lgbm.predict(dtest)
    loss8 = log_loss(stest['is_trade'].values, prob)
    print(loss8)
    del stest
    list_prob += prob.tolist()

    print(len(list_prob))
    del dtest
    del prob
    del lgbm

    print('开始stacking9')
    strain = train1.append([train2, train3, train4, train5, train6, train7, train8, train10])
    stest = train9
    label_all = strain['is_trade'].values
    feature_all = strain.drop(['instance_id', 'user_id', 'time', 'is_trade',
                               'context_timestamp'], axis=1).values
    del strain
    lgbm = trainClassifierXGB(feature_all, label_all)
    del feature_all
    del label_all
    test_all = stest.drop(['instance_id', 'user_id', 'time', 'is_trade',
                           'context_timestamp'], axis=1).values
    dtest = xgb.DMatrix(test_all)
    del test_all
    prob = lgbm.predict(dtest)
    loss9 = log_loss(stest['is_trade'].values, prob)
    print(loss9)
    del stest
    list_prob += prob.tolist()

    print(len(list_prob))
    del dtest
    del prob
    del lgbm

    print('开始stacking10')
    strain = train1.append([train2, train3, train4, train5, train6, train7, train8, train9])
    stest = train10
    label_all = strain['is_trade'].values
    feature_all = strain.drop(['instance_id', 'user_id', 'time', 'is_trade',
                               'context_timestamp'], axis=1).values
    del strain
    lgbm = trainClassifierXGB(feature_all, label_all)
    del feature_all
    del label_all
    test_all = stest.drop(['instance_id', 'user_id', 'time', 'is_trade',
                           'context_timestamp'], axis=1).values
    dtest = xgb.DMatrix(test_all)
    del test_all
    prob = lgbm.predict(dtest)
    loss10 = log_loss(stest['is_trade'].values, prob)
    print(loss10)
    del stest
    list_prob += prob.tolist()

    print(len(list_prob))
    del dtest
    del prob
    del lgbm

    avg_log = (loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9+loss10)/10
    print(avg_log)

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

    df_prob.to_csv('C:/Users/user/Desktop/ijcai_result/prob_xgb_train.txt', sep=' ', index=False)


if __name__=="__main__":

    # stacking_train()
    stacking_test()


    # dataset1_y = dataset1.is_trade
    # dataset1_x = dataset1[new_feature]
    #
    # dataset2_y = dataset2.is_trade
    # dataset2_x = dataset2[new_feature]

    # dataset1_x = dataset1.drop(['instance_id','user_id','user_gender_id','user_occupation_id','time','is_trade','context_timestamp'],axis=1)
    #
    # dataset2_y = dataset2.is_trade
    # dataset2_x = dataset2.drop(['instance_id','user_id','user_gender_id','user_occupation_id','time','is_trade','context_timestamp'],axis=1)
    #
    # dataset12_y = dataset12.is_trade
    # dataset12_x = dataset12.drop(['instance_id','user_id','user_gender_id','user_occupation_id','time',
    #                               'is_trade','context_timestamp'],axis=1)
    #
    # dataset3_preds = dataset3[['instance_id']]
    # dataset3_x = dataset3.drop(['instance_id','user_id','user_gender_id','user_occupation_id','time','context_timestamp'
    #                             ], axis=1)
    #
    # dataset1 = xgb.DMatrix(dataset1_x, label=dataset1_y)
    # dataset2 = xgb.DMatrix(dataset2_x, label=dataset2_y)
    # dataset12 = xgb.DMatrix(dataset12_x, label=dataset12_y)
    # dataset3 = xgb.DMatrix(dataset3_x)
    #
    # print(dataset1_x.shape,dataset2_x.shape)



    # params = {'booster': 'gbtree',
    #           'objective': 'binary:logistic',
    #           'eval_metric': 'logloss',
    #           'gamma': 0.08,
    #           'min_child_weight': 1.1,
    #           'max_depth': 4,
    #           'lambda': 10,
    #           'subsample': 0.7,
    #           'colsample_bytree': 0.7,
    #           'colsample_bylevel': 0.7,
    #           'eta': 0.01,
    #           'tree_method': 'exact',
    #           'seed': 0,
    #           'nthread': 12
    #           }

    # train on dataset1, evaluate on dataset2
    # watchlist = [(dataset1,'train'),(dataset2,'val')]
    # model = xgb.train(params,dataset1,num_boost_round=3500,evals=watchlist,early_stopping_rounds=100)

    # watchlist = [(dataset1,'train')]
    # model = xgb.train(params, dataset1, num_boost_round=1200, evals=watchlist)
    # predict1 = model.predict(dataset2)

    # # predict test set
    # dataset3_preds['predicted_score'] = model.predict(dataset3)
    # print(dataset3_preds['predicted_score'])
    # dataset3_preds.to_csv("C:/Users/user/Desktop/ijcai_result/result_20180404.txt", sep=' ',index=False)

    # feature_score = model.get_fscore()
    # feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
    # fs = []
    # for (key, value) in feature_score:
    #     fs.append("{0},{1}\n".format(key, value))
    #
    # with open('C:/Users/user/Desktop/ijcai_result/xgb_feature_score.csv','w') as f:
    #     f.writelines("feature,score\n")
    #     f.writelines(fs)
