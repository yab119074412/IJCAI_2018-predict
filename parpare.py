#!/usr/bin/python
#-*- coding:utf-8 -*-

import  pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


from scipy import sparse
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

pd.set_option('display.width',1000)

def onehot_encode(dataset1,dataset2):
    enc = OneHotEncoder()
    lb = LabelEncoder()
    feats = ['item_id','context_id','shop_id','last_category_id','item_brand_id','item_city_id','user_gender_id','user_occupation_id',
                  'context_page_id','item_price_level','item_sales_level','item_collected_level','item_pv_level','user_age_level',
                  'user_star_level','shop_review_num_level','shop_star_level']
    for i,feat in enumerate(feats):
        tmp = lb.fit_transform((list(dataset1[feat])+list(dataset2[feat])))
        enc.fit(tmp.reshape(-1,1))
        data1 = enc.transform(lb.transform(dataset1[feat]).reshape(-1, 1))
        data2 = enc.transform(lb.transform(dataset2[feat]).reshape(-1, 1))
        if i==0:
            Data1, Data2 = data1,data2
        else:
            Data1, Data2 = sparse.hstack((Data1,data1)),sparse.hstack((Data2,data2))

    return Data1,Data2



if __name__=="__main__":

    isOnline = True

    dataset1= pd.read_csv('data_csv/final1_encode.csv')
    dataset2 = pd.read_csv('data_csv/final2_encode.csv')
    dataset3 = pd.read_csv('data_csv/final3_encode.csv')

    dataset1.fillna(value=0, inplace=True)
    dataset2.fillna(value=0, inplace=True)
    dataset3.fillna(value=0, inplace=True)

    dataset1.drop_duplicates(inplace=True)
    dataset2.drop_duplicates(inplace=True)
    dataset3.drop_duplicates(inplace=True)

    df_all = pd.concat([dataset1, dataset2], axis=0)

    train_sta_xgb = pd.read_csv('C:/Users/user/Desktop/ijcai_result/prob_xgb_train.txt', sep=' ')
    df_all_train = pd.merge(df_all, train_sta_xgb, how='left', on='instance_id')

    test_sta_xgb = pd.read_csv('C:/Users/user/Desktop/ijcai_result/prob_xgb_test.txt', sep=' ')
    df_all_test = pd.merge(dataset3, test_sta_xgb, how='left', on='instance_id')

    data1_x,data2_x = onehot_encode(df_all_train,df_all_test)

    drop=['instance_id','user_id','time','is_trade','context_timestamp','item_id','context_id','shop_id','last_category_id',
          'item_brand_id','item_city_id','user_gender_id','user_occupation_id','context_page_id','item_price_level',
          'item_sales_level','item_collected_level','item_pv_level','user_age_level','user_star_level',
          'shop_review_num_level','shop_star_level']

    test_drop = ['instance_id', 'user_id', 'time', 'context_timestamp', 'item_id', 'context_id', 'shop_id',
            'last_category_id',
            'item_brand_id', 'item_city_id', 'user_gender_id', 'user_occupation_id', 'context_page_id',
            'item_price_level',
            'item_sales_level', 'item_collected_level', 'item_pv_level', 'user_age_level', 'user_star_level',
            'shop_review_num_level', 'shop_star_level']

    df_all_train_y = df_all_train.is_trade
    df_all_train_x = df_all_train.drop(drop,axis=1)

    train_x, val_x, train_y,val_y = train_test_split(df_all_train_x,df_all_train_y,test_size=0.3,random_state=0)

    test_preds = dataset3[['instance_id']]
    test_x = df_all_test.drop(test_drop,axis=1)

    model_lgb = lgb.LGBMClassifier(objective='binary',
                            num_leaves=24,
                            max_depth=3,
                            learning_rate=0.01,
                            n_estimators=13,
                            colsample_bytree = 1.0)


    if  isOnline == False:

         model_lgb.fit(train_x, train_y, eval_metric='logloss')

         val_predict = model_lgb.predict_proba(val_x)[:, 1]
         lgb_val_log_loss = log_loss(val_y,val_predict)
         print('lgb log_loss: %.5f' % lgb_val_log_loss)

         train_leaves = model_lgb.apply(train_x)
         val_leaves = model_lgb.apply(val_x)
         (train_rows, cols) = train_leaves.shape
         all_leaves = np.concatenate((train_leaves, val_leaves), axis=0)

         lgbenc = OneHotEncoder()
         X_trans = lgbenc.fit_transform(all_leaves)
         lr = LogisticRegression(penalty='l1',C=3)
         lr.fit(X_trans[:train_rows,:] ,train_y)
         val_predict_lgb_lr = lr.predict_proba(X_trans[train_rows:, :])[:, 1]
         lgb_lr_val_log_loss = log_loss(val_y, val_predict_lgb_lr)
         print('基于lgb特征编码后的LR log_loss: %.5f' % lgb_lr_val_log_loss)

         # lr + lgb 组合特征
         # train_ext = sparse.hstack([X_trans[:train_rows,:] , data1_x.iloc[:train_rows,:]])
         # val_ext = sparse.hstack([X_trans[train_rows:,:], data1_x.iloc[train_rows:,:]])
         # lr.fit(train_ext, train_y)
         # y_pred_lgb_lr2 = lr.predict_proba(val_ext)[:, 1]
         # lgb_lr_val_log_loss2 = log_loss(val_y, y_pred_lgb_lr2)
         # print('基于组合特征的LR log_loss: %.5f' % lgb_lr_val_log_loss2)

    else:

        model_lgb.fit(df_all_train_x, df_all_train_y, eval_metric='logloss')

        train_leaves = model_lgb.apply(df_all_train_x)
        test_leaves = model_lgb.apply(test_x)
        (train_rows, cols) = train_leaves.shape
        all_leaves = np.concatenate((train_leaves, test_leaves), axis=0)

        lgbenc = OneHotEncoder()
        X_trans = lgbenc.fit_transform(all_leaves)

        lr = LogisticRegression(penalty='l1',C=3)
        train_ext = sparse.hstack([X_trans[:train_rows, :], data1_x])
        test_ext = sparse.hstack([X_trans[train_rows:, :],  data2_x])

        lr.fit(train_ext, df_all_train_y)

        test_preds['predicted_score'] = lr.predict_proba(test_ext)[:, 1]
        print(test_preds['predicted_score'])
        test_preds_1 = test_preds.iloc[18371:,:]
        test_preds_1.to_csv("C:/Users/user/Desktop/ijcai_result/result_201804019_lr.txt", sep=' ', index=False)