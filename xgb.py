#!/usr/bin/python
#-*- coding:utf-8 -*-

import  pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy import sparse
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.width',1000)


dataset1 = pd.read_csv('data_csv/final1_encode.csv')
dataset2 = pd.read_csv('data_csv/final2_encode.csv')
dataset3 = pd.read_csv('data_csv/final3_encode.csv')


dataset1.fillna(value=0, inplace=True)
dataset2.fillna(value=0, inplace=True)
dataset3.fillna(value=0, inplace=True)


dataset1.drop_duplicates(inplace=True)
dataset2.drop_duplicates(inplace=True)
dataset3.drop_duplicates(inplace=True)

# gender_dummies = pd.get_dummies(dataset1.user_gender_id)
# gender_dummies.columns = ['gender'+str(i+1) for i in range(gender_dummies.shape[1])]
# dataset1 = pd.concat([dataset1,gender_dummies],axis=1)
#
# gender_dummies = pd.get_dummies(dataset2.user_gender_id)
# gender_dummies.columns = ['gender'+str(i+1) for i in range(gender_dummies.shape[1])]
# dataset2 = pd.concat([dataset2,gender_dummies],axis=1)
#
# gender_dummies = pd.get_dummies(dataset3.user_gender_id)
# gender_dummies.columns = ['gender'+str(i+1) for i in range(gender_dummies.shape[1])]
# dataset3 = pd.concat([dataset3,gender_dummies],axis=1)
#
# occupation_dummies = pd.get_dummies(dataset1.occupation0)
# occupation_dummies.columns = ['occupation_id'+str(i+1) for i in range(occupation_dummies.shape[1])]
# dataset1 = pd.concat([dataset1,occupation_dummies],axis=1)
#
# occupation_dummies = pd.get_dummies(dataset2.occupation0)
# occupation_dummies.columns = ['occupation_id'+str(i+1) for i in range(occupation_dummies.shape[1])]
# dataset2 = pd.concat([dataset2,occupation_dummies],axis=1)
#
# occupation_dummies = pd.get_dummies(dataset3.occupation0)
# occupation_dummies.columns = ['occupation_id'+str(i+1) for i in range(occupation_dummies.shape[1])]
# dataset3 = pd.concat([dataset3,occupation_dummies],axis=1)
#
# star_dummies = pd.get_dummies(dataset1.star0)
# star_dummies.columns = ['star_level'+str(i+1) for i in range(star_dummies.shape[1])]
# dataset1 = pd.concat([dataset1,star_dummies],axis=1)
#
# star_dummies = pd.get_dummies(dataset2.star0)
# star_dummies.columns = ['star_level'+str(i+1) for i in range(star_dummies.shape[1])]
# dataset2 = pd.concat([dataset2,star_dummies],axis=1)
#
# star_dummies = pd.get_dummies(dataset3.star0)
# star_dummies.columns = ['star_level'+str(i+1) for i in range(star_dummies.shape[1])]
# dataset3 = pd.concat([dataset3,star_dummies],axis=1)

# Features=pd.read_csv('C:/Users/user/Desktop/ijcai_result/xgb_feature_score.csv')
# feature=Features['feature']
# new_feature=list(feature.iloc[:104])

# all_data = pd.concat([dataset1,dataset2],axis=0)
#
# all_data_x = all_data.drop(['instance_id','user_id','time','is_trade','context_timestamp'],axis=1)
# all_data_y = all_data.is_trade


# train_x,text_x,train_y,test_y = train_test_split(all_data_x,all_data_y,test_size=0.3,random_state=0)

dataset1_y = dataset1.is_trade
dataset1_x = dataset1.drop(['instance_id','user_id','time','context_timestamp','is_trade',
                            'pv_transfer_rate',
                            'gender_price_transfer_rate',
                            'star_occupation_price_transfer_rate',
                            'star_star_transfer_rate',
                            'user_gender_transfer_rate',
                            'category_transfer_rate',
                            'this_day_user_constant_click_times',
                            'this_day_user_click_same_brand_times',
                            'gender_star_price_transfer_rate',
                            'this_day_user_click_same_shop_times',
                            'maybe',
                            'price_transfer_rate',
                            'user_hour_click_times',
                            'item_category_list2',
                            'star_price_transfer_rate',
                            'len_item_category'
                            ],axis=1)


dataset2_y = dataset2.is_trade
dataset2_x = dataset2.drop(['instance_id','user_id','time','context_timestamp','is_trade',
                            'pv_transfer_rate',
                            'gender_price_transfer_rate',
                            'star_occupation_price_transfer_rate',
                            'star_star_transfer_rate',
                            'user_gender_transfer_rate',
                            'category_transfer_rate',
                            'this_day_user_constant_click_times',
                            'this_day_user_click_same_brand_times',
                            'gender_star_price_transfer_rate',
                            'this_day_user_click_same_shop_times',
                            'maybe',
                            'price_transfer_rate',
                            'user_hour_click_times',
                            'item_category_list2',
                            'star_price_transfer_rate',
                            'len_item_category'
                            ],axis=1)

dataset3_preds = dataset3[['instance_id']]
dataset3_x = dataset3.drop(['instance_id','user_id','time','context_timestamp','is_trade',
                            'star0', 'occupation0', 'user_gender_id'],axis=1)

dataset12 = pd.concat([dataset1,dataset2],axis=0)
dataset12_y = dataset12.is_trade
dataset12_x = dataset12.drop(['instance_id','user_id','time','context_timestamp','is_trade',
                            'star0', 'occupation0', 'user_gender_id'],axis=1)

train_data = xgb.DMatrix(dataset1_x, label=dataset1_y)
val_data = xgb.DMatrix(dataset2_x, label=dataset2_y)
test_data = xgb.DMatrix(dataset3_x)
print(dataset1_x.shape,dataset2_x.shape,dataset3_x.shape)

dataset12 = xgb.DMatrix(dataset12_x, label=dataset12_y)

# train_data = xgb.DMatrix(train_x, label=train_y)
# val_data = xgb.DMatrix(text_x, label=test_y)
# test_data = xgb.DMatrix(dataset3_x)
# print(dataset3_x.shape,all_data_x.shape)

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
          'scale_pos_weight':1,
          'nthread': 12
          }

# train on dataset1, evaluate on dataset2

watchlist = [(train_data,'train'),(val_data,'val')]
model = xgb.train(params,train_data,num_boost_round=3500,evals=watchlist,early_stopping_rounds=100)
dataset2_preds['predicted_score'] = model.predict(val_data)
print(log_loss(dataset2_y,dataset2_preds['predicted_score']))
dataset2_preds.to_csv("C:/Users/user/Desktop/result_data2.txt", sep=' ',index=False)

# dataset1 = xgb.DMatrix(dataset1_x, label=dataset1_y)
# dataset2 = xgb.DMatrix(dataset2_x, label=dataset2_y)
# dataset12 = xgb.DMatrix(dataset12_x, label=dataset12_y)
# dataset3 = xgb.DMatrix(dataset3_x)
# print(dataset1_x.shape,dataset2_x.shape,dataset3_x.shape)


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
#           'scale_pos_weight':1,
#           'nthread': 12
#           }
#
# # train on dataset1, evaluate on dataset2
# watchlist = [(dataset1,'train'),(dataset2,'val')]
# model = xgb.train(params,dataset1,num_boost_round=3500,evals=watchlist,early_stopping_rounds=100)

# watchlist = [(dataset12,'train')]
# model = xgb.train(params, dataset12, num_boost_round=1500, evals=watchlist)


# model_lgb = lgb.LGBMClassifier(objective='binary',
#                         num_leaves=64,
#                         max_depth=6,
#                         learning_rate=0.01,
#                         n_estimators=1000,
#                         colsample_bytree = 1.0)
#
# model_lgb.fit(dataset1_x, dataset1_y,eval_metric='logloss')
# predict2 = model_lgb.predict_proba(dataset2_x)[:, 1]
#
# print(log_loss(dataset2_y,predict2))

# predict test set
# dataset3_preds['predicted_score'] = model.predict(test_data)
# print(dataset3_preds['predicted_score'])
# submit = dataset3_preds.iloc[18371:, :]
# submit.to_csv("C:/Users/user/Desktop/ijcai_result/result_20180421.txt", sep=' ',index=False)

# feature_score = model.get_fscore()
# feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
# fs = []
# for (key, value) in feature_score:
#     fs.append("{0},{1}\n".format(key, value))
#
# with open('C:/Users/user/Desktop/ijcai_result/xgb_feature_score.csv','w') as f:
#     f.writelines("feature,score\n")
#     f.writelines(fs)
