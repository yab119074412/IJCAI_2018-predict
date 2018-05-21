#!/usr/bin/python
#-*- coding:utf-8 -*-

import  pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

from scipy import sparse
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

pd.set_option('display.width',1000)

def onehot_encode(dataset1,dataset2,dataset3):
    enc = OneHotEncoder()
    lb = LabelEncoder()
    feats = ['item_id','context_id','shop_id','last_category_id','item_brand_id','item_city_id','user_gender_id','user_occupation_id',
                  'context_page_id','item_price_level','item_sales_level','item_collected_level','item_pv_level','user_age_level',
                  'user_star_level','shop_review_num_level','shop_star_level']
    for i,feat in enumerate(feats):
        tmp = lb.fit_transform((list(dataset1[feat])+list(dataset2[feat])+list(dataset3[feat])))
        enc.fit(tmp.reshape(-1,1))
        data1 = enc.transform(lb.transform(dataset1[feat]).reshape(-1, 1))
        print(data1)
        data2 = enc.transform(lb.transform(dataset2[feat]).reshape(-1, 1))
        data3 = enc.transform(lb.transform(dataset3[feat]).reshape(-1, 1))
        if i==0:
            Data1, Data2, Data3 = data1,data2,data3
        else:
            Data1, Data2, Data3 = sparse.hstack((Data1,data1)),sparse.hstack((Data2,data2)),sparse.hstack((Data3,data3))

    return Data1,Data2,Data3

dataset1= pd.read_csv('data_csv/final1_encode.csv')
dataset2 = pd.read_csv('data_csv/final2_encode.csv')
dataset3 = pd.read_csv('data_csv/final3_encode.csv')

dataset1.fillna(value=0, inplace=True)
dataset2.fillna(value=0, inplace=True)
dataset3.fillna(value=0, inplace=True)

dataset1.drop_duplicates(inplace=True)
dataset2.drop_duplicates(inplace=True)
dataset3.drop_duplicates(inplace=True)

dataset1_y = dataset1.is_trade
dataset1_x = dataset1.drop(['instance_id','user_id','time',
                            'is_trade','context_timestamp'],axis=1)

dataset2_y = dataset2.is_trade
dataset2_x = dataset2.drop(['instance_id','user_id','time',
                            'is_trade','context_timestamp'],axis=1)

dataset12 = pd.concat([dataset1, dataset2], axis=0)
dataset12_y = dataset12.is_trade
dataset12_x = dataset12.drop(['instance_id','user_id','time',
                            'is_trade','context_timestamp'],axis=1)

dataset3_preds = dataset3[['instance_id']]
dataset3_x = dataset3.drop(['instance_id','user_id','time','context_timestamp'], axis=1)

# data1_x,data2_x,data3_x = onehot_encode(dataset1_x,dataset2_x,dataset3_x)

# lr = LogisticRegression(penalty='l1')
#
# lr.fit(data1_x,dataset1_y)
# prob_data2 = lr.predict_proba(data2_x)[:,1]
# prob_data3 = lr.predict_proba(data3_x)[:,1]
# print(log_loss(dataset2_y,prob_data2))


drop=['this_day_city_click_times','gender_shop_transfer_rate',
      'age_sales_transfer_rate','this_day_star_click_item_times','occupation_price_transfer_rate','this_day_gender_click_item_times',
      'this_day_star_click_brand_times','star_collected_transfer_rate','this_day_gender_click_brand_times','age_city_transfer_rate',
      'gender_category_transfer_rate','gender_occupation_category_transfer_rate','star_price_transfer_rate','occupation_star_transfer_rate',
      'this_day_user_click_same_category_times','shop_market_share_on_brand','this_day_hour_item_is_click_times','age_star_item_transfer_rate',
      'context_page_transfer_rate','gender_pv_transfer_rate','this_day_brand_click_times','star_category_transfer_rate',
      'star_occupation_shop_transfer_rate','this_day_occupation_click_brand_times','this_day_age_click_shop_times','age_category_transfer_rate',
      'occupation_sales_transfer_rate','user_transfer_rate','gender_collected_transfer_rate','age_occupation_item_transfer_rate',
      'this_day_occupation_click_category_times','this_day_occupation_click_item_times','gender_shop_star_transfer_rate',
      'umaybe','gender_star_item_transfer_rate','user_click_same_item_times','gender_occupation_price_transfer_rate','this_day_star_click_shop_times',
      'gender_sales_transfer_rate','this_day_occupation_click_shop_times','occupation_pv_transfer_rate','this_day_gender_click_shop_times',
      'star_occupation_item_transfer_rate','user_click_brand_times','city_transfer_rate','gender_city_transfer_rate',
      'gender_price_transfer_rate','this_day_user_click_same_city_times','user_click_same_shop_times','shop_star_transfer_rate','this_day_context_page_click_times',
      'user_click_same_shop_item_times','this_day_user_click_same_page_times','shop_user_share_on_brand','user_star_transfer_rate',
      'collected_transfer_rate','user_click_city_times','shop_brand_trade_times_Umb','this_day_category_click_times',
      'brand_market_share_with_shop','cmaybe','shop_review_num_transfer_rate','category_transfer_rate','item_hour_is_tarde_times',
      'brand_user_share_with_shop','pv_transfer_rate','user_click_category_times','user_day_hour_minute_click_times',
      'sales_transfer_rate','user_click_price_times','user_context_page_buy_times','maybe','this_day_user_click_same_brand_times',
     'this_day_user_click_same_shop_times','user_day_hour_click_times','this_day_user_click_same_item_times','this_day_user_constant_click_times',
      'user_gender_transfer_rate','price_transfer_rate','user_age_transfer_rate','user_hour_click_times','bmaybe']

train_y = dataset1.is_trade
train_x = dataset1_x.drop(drop,axis=1)

val_y = dataset2.is_trade
val_x = dataset2_x.drop(drop,axis=1)

dataset12 = pd.concat([dataset1, dataset2], axis=0)
dataset12_y = dataset12.is_trade
dataset12_x = dataset12_x.drop(drop,axis=1)

test_preds = dataset3[['instance_id']]
test_x = dataset3_x.drop(drop,axis=1)

dataset1 = xgb.DMatrix(train_x, label=train_y)
dataset2 = xgb.DMatrix(val_x , label=val_y)
dataset12 = xgb.DMatrix(dataset12_x, label=dataset12_y)
dataset3 = xgb.DMatrix(test_x)
print(train_x.shape,val_x.shape,test_x.shape)


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

# train on dataset1, evaluate on dataset2
watchlist = [(dataset1,'train'),(dataset2,'val')]
model = xgb.train(params,dataset1,num_boost_round=3500,evals=watchlist,early_stopping_rounds=100)

# watchlist = [(dataset12,'train')]
# model = xgb.train(params, dataset12, num_boost_round=1600, evals=watchlist)


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
# dataset3_preds['predicted_score'] = model.predict(dataset3)
# print(dataset3_preds['predicted_score'])
# dataset3_preds_1 = dataset3_preds.iloc[18371:,:]
# dataset3_preds_1.to_csv("C:/Users/user/Desktop/ijcai_result/result_201804018_xgb.txt", sep=' ',index=False)

# feature_score = model.get_fscore()
# feature_score = sorted(feature_score.items(), key=lambda x: x[1], reverse=True)
# fs = []
# for (key, value) in feature_score:
#     fs.append("{0},{1}\n".format(key, value))
#
# with open('C:/Users/user/Desktop/ijcai_result/xgb_feature_score2.csv','w') as f:
#     f.writelines("feature,score\n")
#     f.writelines(fs)


