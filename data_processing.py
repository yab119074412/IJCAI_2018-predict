#!/usr/bin/python
#-*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import random
import time
import scipy.special as special
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.width',1000)


class HyperParam(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample_from_beta(self, alpha, beta, num, imp_upperbound):
        #产生样例数据
        sample = np.random.beta(alpha, beta, num)
        I = []
        C = []
        for click_ratio in sample:
            imp = random.random() * imp_upperbound
            #imp = imp_upperbound
            click = imp * click_ratio
            I.append(imp)
            C.append(click)
        return pd.Series(I), pd.Series(C)

    def update_from_data_by_FPI(self, tries, success, iter_num, epsilon):
        #更新策略
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, tries, success, alpha, beta):
        #迭代函数
        sumfenzialpha = 0.0
        sumfenzibeta = 0.0
        sumfenmu = 0.0
        sumfenzialpha = (special.digamma(success+alpha) - special.digamma(alpha)).sum()
        sumfenzibeta = (special.digamma(tries-success+beta) - special.digamma(beta)).sum()
        sumfenmu = (special.digamma(tries+alpha+beta) - special.digamma(alpha+beta)).sum()

        return alpha*(sumfenzialpha/sumfenmu), beta*(sumfenzibeta/sumfenmu)

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

def star(x):
    if x==-1 | x==3000:
        return 1
    elif x==3009 | x==3010:
        return 2
    else:
        return 3

def occupation(x):
    if x==-1 | x==2003:
        return 1
    elif x==2002:
        return 2
    else:
        return 3


def base_process(data):
    lbl = preprocessing.LabelEncoder()
    data['len_item_category']=data['item_category_list'].apply(lambda x:len(x.split(';')))
    data['len_item_property']=data['item_property_list'].apply(lambda x:len(x.split(';')))
    for i in range(1, 3):
        data['item_category_list' + str(i)] = lbl.fit_transform(data['item_category_list'].map(
            lambda x: str(str(x).split(';')[i]) if len(str(x).split(';')) > i else ''))

    for col in ['item_id','item_brand_id','item_city_id','user_id','shop_id','context_id']:
        data[col] = lbl.fit_transform(data[col])

    data['occupation0'] = data['user_occupation_id'].apply(occupation)
    data['star0'] = data['user_star_level'].apply(star)

    data = data.drop(['user_occupation_id','user_star_level'],axis=1)
    return data

def last_category(x):
    l = len(x)
    return x[l-1]


def split_data():
    train_data = pd.read_table('data_csv/round1_ijcai_18_train_20180301.txt', sep=' ')
    train_data.drop_duplicates(inplace=True)

    test_a = pd.read_csv('data_csv/round1_ijcai_18_test_a_20180301.txt', sep=' ')
    test_a['item_category_list_1'] = test_a['item_category_list'].apply(lambda x:x.split(';'))
    test_a['last_category_id'] = test_a['item_category_list_1'].apply(last_category)
    test_b = pd.read_csv('data_csv/round1_ijcai_18_test_b_20180418.txt', sep=' ')
    test = pd.concat([test_a,test_b],axis=0)
    
    data = pd.concat([train_data,test],axis=0)
    data.reset_index(drop=True, inplace=True)
    data = convert_data(data)
    
    print('数据预处理')
    data = base_process(data)
    
    dataset3 = data.loc[data.day == 25]
    feature3 = data.loc[(data.day < 25) & (data.day >19)]
    dataset2 = data.loc[data.day == 24]
    feature2 = data.loc[(data.day < 24) & (data.day >18)]
    dataset1 = data.loc[data.day == 23]
    feature1 = data.loc[data.day <23]
    
    return dataset3,feature3,dataset2,feature2,dataset1,feature1

# 提取预测数据当天特征
def get_label_feature(dataset):
    t=dataset[['user_id','day']]
    t['this_day_user_click_times']=1
    t=t.groupby(['user_id','day']).agg('sum').reset_index()
    label_feature=pd.merge(dataset,t,on=['user_id','day'],how='left')

    # t = dataset[['user_id','item_id','shop_id','day','hour','minute']]
    # t['this_day_user_one_minute_click_times'] = 1
    # t = t.groupby(['user_id','item_id','shop_id','day','hour','minute']).agg('sum').reset_index()
    # label_feature = pd.merge(dataset, t, on=['user_id','item_id','shop_id','day','hour','minute'], how='left')



    t1=dataset[['user_id','day','hour']]
    t1['this_day_hour_click_times']=1
    t1=t1.groupby(['user_id','day','hour']).agg('sum').reset_index()
    label_feature=pd.merge(label_feature,t1,on=['user_id','day','hour'],how='left')

    t2=dataset[['user_id','item_id']]
    t2['this_day_user_click_same_item_times'] = 1
    t2 = t2.groupby(['user_id', 'item_id']).agg('sum').reset_index()
    label_feature=pd.merge(label_feature,t2,on=['user_id','item_id'],how='left')

    t3=dataset[['user_id','shop_id']]
    t3['this_day_user_click_same_shop_times'] = 1
    t3 = t3.groupby(['user_id', 'shop_id']).agg('sum').reset_index()
    label_feature = pd.merge(label_feature, t3, on=['user_id','shop_id'], how='left')

    t6=dataset[['user_id','item_brand_id']]
    t6['this_day_user_click_same_brand_times']=1
    t6=t6.groupby(['user_id','item_brand_id']).agg('sum').reset_index()
    label_feature = pd.merge(label_feature, t6, on=['user_id','item_brand_id'], how='left')

    t9=dataset[['user_id','item_category_list1']]
    t9['this_day_user_click_same_category_times']=1
    t9=t9.groupby(['user_id','item_category_list1']).agg('sum').reset_index()
    label_feature = pd.merge(label_feature, t9, on=['user_id','item_category_list1'], how='left')

    t10=dataset[['user_id','day','hour','minute']]
    t10['this_day_user_constant_click_times']=1
    t10=t10.groupby(['user_id','day','hour','minute']).agg('sum').reset_index()
    label_feature = pd.merge(label_feature, t10, on=['user_id','day','hour','minute'], how='left')

    t14 = dataset[['user_gender_id', 'item_category_list1', 'day']]
    t14['this_day_gender_click_category_times'] = 1
    t14 = t14.groupby(['user_gender_id', 'item_category_list1', 'day']).agg('sum').reset_index()
    label_feature = pd.merge(label_feature, t14, on=['user_gender_id', 'item_category_list1', 'day'], how='left')

    t16 = dataset[['user_age_level', 'item_id', 'day']]
    t16['this_day_age_click_item_times'] = 1
    t16 = t16.groupby(['user_age_level', 'item_id', 'day']).agg('sum').reset_index()
    label_feature = pd.merge(label_feature, t16, on=['user_age_level', 'item_id', 'day'], how='left')

    t17 = dataset[['user_age_level', 'item_brand_id', 'day']]
    t17['this_day_age_click_brand_times'] = 1
    t17 = t17.groupby(['user_age_level', 'item_brand_id', 'day']).agg('sum').reset_index()
    label_feature = pd.merge(label_feature, t17, on=['user_age_level', 'item_brand_id', 'day'], how='left')

    t19 = dataset[['user_age_level', 'item_category_list1', 'day']]
    t19['this_day_age_click_category_times'] = 1
    t19 = t19.groupby(['user_age_level', 'item_category_list1', 'day']).agg('sum').reset_index()
    label_feature = pd.merge(label_feature, t19, on=['user_age_level', 'item_category_list1', 'day'], how='left')

    t21 = dataset[['occupation0', 'item_id', 'day']]
    t21['this_day_occupation_click_item_times'] = 1
    t21 = t21.groupby(['occupation0', 'item_id', 'day']).agg('sum').reset_index()
    label_feature = pd.merge(label_feature, t21, on=['occupation0', 'item_id', 'day'], how='left')

    t22 = dataset[['occupation0', 'item_brand_id', 'day']]
    t22['this_day_occupation_click_brand_times'] = 1
    t22 = t22.groupby(['occupation0', 'item_brand_id', 'day']).agg('sum').reset_index()
    label_feature = pd.merge(label_feature, t22, on=['occupation0', 'item_brand_id', 'day'], how='left')

    t26 = dataset[['star0', 'item_id', 'day']]
    t26['this_day_star_click_item_times'] = 1
    t26 = t26.groupby(['star0', 'item_id', 'day']).agg('sum').reset_index()
    label_feature = pd.merge(label_feature, t26, on=['star0', 'item_id', 'day'], how='left')

    t27 = dataset[['star0', 'item_brand_id', 'day']]
    t27['this_day_star_click_brand_times'] = 1
    t27 = t27.groupby(['star0', 'item_brand_id', 'day']).agg('sum').reset_index()
    label_feature = pd.merge(label_feature, t27, on=['star0', 'item_brand_id', 'day'], how='left')

    t30 = dataset[['star0', 'shop_id', 'day']]
    t30['this_day_star_click_shop_times'] = 1
    t30 = t30.groupby(['star0', 'shop_id', 'day']).agg('sum').reset_index()
    label_feature = pd.merge(label_feature, t30, on=['star0', 'shop_id', 'day'], how='left')

    t31=dataset[['day','item_id']]
    t31['this_day_item_is_click_times']=1
    t31 = t31.groupby(['day','item_id']).agg('sum').reset_index()
    label_feature = pd.merge(label_feature, t31, on=['day','item_id'], how='left')

    t32=dataset[['day','hour','item_id']]
    t32['this_day_hour_item_is_click_times']=1
    t32=t32.groupby(['day','hour','item_id']).agg('sum').reset_index()
    label_feature = pd.merge(label_feature, t32, on=['day','hour','item_id'], how='left')

    t37=dataset[['occupation0','day','hour']]
    t37['this_day_occupation_hour_click_times']=1
    t37=t37.groupby(['occupation0','day','hour']).agg('sum').reset_index()
    label_feature=pd.merge(label_feature,t37,on=['occupation0','day','hour'],how='left')

    return label_feature

# 根据划窗提取历史数据单特征
def get_one_feature(feature,label_feature):
    t = feature[['user_id']]
    t['user_click_total_times'] = 1
    t = t.groupby('user_id').agg('sum').reset_index()
    dataset = pd.merge(label_feature, t, on='user_id', how='left')
    dataset.user_click_total_times = dataset.user_click_total_times.replace(np.nan,0)

    t1 = feature[['user_id', 'is_trade']]
    t1['user_buy_times'] = t1[['is_trade']]
    t1 = t1.groupby('user_id')['user_buy_times'].agg('sum').reset_index()
    dataset = pd.merge(dataset, t1, on='user_id', how='left')
    dataset.user_buy_times = dataset.user_buy_times.replace(np.nan, 0)

    # hyper1 = HyperParam(1, 1)
    # hyper1.update_from_data_by_FPI(dataset['user_click_total_times'].values,dataset['user_buy_times'].values, 1000,
    #                                0.0001)
    # print('hyper1:',hyper1.alpha,hyper1.beta)
    alpha1 =0.0769474405685425
    beta1 = 9.12774893416997

    # alpha1 = 0.05498294291193993
    # beta1  = 7.387901428754157

    # alpha1 = 0.07053205066881155
    # beta1 = 8.186153702658457

    # dataset['user_transfer_rate'] = dataset['user_buy_times'].apply(lambda x: float(x) + hyper1.alpha) / \
    #                                      dataset['user_click_total_times'].apply(lambda x: float(x) + hyper1.alpha + hyper1.beta)
    # dataset.user_transfer_rate = dataset.user_transfer_rate.replace(np.nan, hyper1.alpha / (hyper1.alpha + hyper1.beta))

    dataset['user_transfer_rate'] = dataset['user_buy_times'].apply(lambda x: float(x) + alpha1)\
                                    / dataset['user_click_total_times'].apply(lambda x: float(x) + alpha1 + beta1)
    dataset.user_transfer_rate = dataset.user_transfer_rate.replace(np.nan, alpha1 / (alpha1 + beta1))


    t2 = feature[['user_gender_id']]
    t2['user_gender_click_times'] = 1
    t2 = t2.groupby('user_gender_id').agg('sum').reset_index()
    dataset = pd.merge(dataset, t2, on='user_gender_id', how='left')

    t3 = feature[['user_gender_id', 'is_trade']]
    t3['user_gender_trade_times'] = t3['is_trade']
    t3 = t3.groupby('user_gender_id')['user_gender_trade_times'].agg('sum').reset_index()
    dataset = pd.merge(dataset, t3, on='user_gender_id', how='left')

    dataset['user_gender_transfer_rate'] = dataset['user_gender_trade_times'] / dataset['user_gender_click_times']

    t6 = feature[['user_age_level']]
    t6['user_age_click_times'] = 1
    t6 = t6.groupby('user_age_level').agg('sum').reset_index()
    dataset = pd.merge(dataset, t6, on='user_age_level', how='left')

    t7 = feature[['user_age_level', 'is_trade']]
    t7['user_age_trade_times'] = t7['is_trade']
    t7 = t7.groupby('user_age_level')['user_age_trade_times'].agg(lambda x: sum(x == 1)).reset_index()
    dataset = pd.merge(dataset, t7, on='user_age_level', how='left')

    dataset['user_age_transfer_rate'] = dataset['user_age_trade_times'] / dataset['user_age_click_times']

    t8 = feature[['star0']]
    t8['user_star_click_times'] = 1
    t8 = t8.groupby('star0').agg('sum').reset_index()
    dataset = pd.merge(dataset, t8, on='star0', how='left')

    t9 = feature[['star0', 'is_trade']]
    t9['user_star_trade_times'] = t9['is_trade']
    t9 = t9.groupby('star0')['user_star_trade_times'].agg('sum').reset_index()
    dataset = pd.merge(dataset, t9, on='star0', how='left')

    dataset['user_star_transfer_rate'] = dataset['user_star_trade_times'] / dataset['user_star_click_times']

    t10 = feature[['item_id']]
    t10['item_is_click_total_times'] = 1
    t10 = t10.groupby('item_id').agg('sum').reset_index()
    dataset = pd.merge(dataset, t10, on='item_id', how='left')
    dataset.item_is_click_total_times = dataset.item_is_click_total_times.replace(np.nan, 0)

    t11 = feature[['item_id', 'is_trade']]
    t11['item_is_trade_times'] = t11['is_trade']
    t11 = t11.groupby('item_id')['item_is_trade_times'].agg('sum').reset_index()
    dataset = pd.merge(dataset, t11, on='item_id', how='left')
    dataset.item_is_trade_times = dataset.item_is_trade_times.replace(np.nan, 0)


    # hyper2 = HyperParam(1, 1)
    # hyper2.update_from_data_by_FPI(dataset['item_is_click_total_times'].values, dataset['item_is_trade_times'].values, 1000,
    #                                0.0001)
    # print('hyper2:', hyper2.alpha, hyper2.beta)
    #
    alpha2 =1.4270011322449327
    beta2 = 71.27269357067141

    # alpha2 = 1.3950118560565037
    # beta2 = 67.76543715589267

    # alpha2 = 1.4406905436867319
    # beta2 = 68.06188442339152
    # dataset['item_quality'] = dataset['item_is_trade_times'].apply(lambda x: float(x) + hyper2.alpha) / \
    #                           dataset['item_is_click_total_times'].apply(lambda x: float(x) + hyper2.alpha +
    #                                                                                hyper2.beta)
    # dataset.item_quality = dataset.item_quality.replace(np.nan, hyper2.alpha / (hyper2.alpha + hyper2.beta))

    dataset['item_quality'] = dataset['item_is_trade_times'].apply(lambda x: float(x) + alpha2) / \
                              dataset['item_is_click_total_times'].apply(lambda x: float(x) + alpha2 +beta2)
    dataset.item_quality = dataset.item_quality.replace(np.nan, alpha2 / (alpha2 + beta2))

    t12 = feature[['item_category_list1']]
    t12['same_category_is_click_times'] = 1
    t12 = t12.groupby('item_category_list1').agg('sum').reset_index()
    dataset = pd.merge(dataset, t12, on='item_category_list1', how='left')

    t13 = feature[['item_category_list1', 'is_trade']]
    t13['same_category_is_buy_times'] = t13['is_trade']
    t13 = t13.groupby('item_category_list1')['same_category_is_buy_times'].agg('sum').reset_index()
    dataset = pd.merge(dataset, t13, on='item_category_list1', how='left')

    dataset['category_transfer_rate'] = dataset['same_category_is_buy_times'] / dataset['same_category_is_click_times']

    t14 = feature[['item_brand_id']]
    t14['item_brand_click_total_times'] = 1
    t14 = t14.groupby('item_brand_id').agg('sum').reset_index()
    dataset = pd.merge(dataset, t14, on='item_brand_id', how='left')

    t15 = feature[['item_brand_id', 'is_trade']]
    t15['item_brand_trade_total_times'] = t15['is_trade']
    t15 = t15.groupby('item_brand_id')['item_brand_trade_total_times'].agg('sum').reset_index()
    dataset = pd.merge(dataset, t15, on='item_brand_id', how='left')
    dataset['brand_transfer_rate'] = dataset['item_brand_trade_total_times'] / dataset['item_brand_click_total_times']

    t16 = feature[['item_city_id']]
    t16['item_city_click_total_times'] = 1
    t16 = t16.groupby('item_city_id').agg('sum').reset_index()
    dataset = pd.merge(dataset, t16, on='item_city_id', how='left')

    t17 = feature[['item_city_id', 'is_trade']]
    t17['item_city_trade_total_times'] = t17['is_trade']
    t17 = t17.groupby('item_city_id')['item_city_trade_total_times'].agg('sum').reset_index()
    dataset = pd.merge(dataset, t17, on='item_city_id', how='left')
    dataset['city_transfer_rate'] = dataset['item_city_trade_total_times'] / dataset['item_city_click_total_times']

    t18 = feature[['item_price_level']]
    t18['item_price_click_total_times'] = 1
    t18 = t18.groupby('item_price_level').agg('sum').reset_index()
    dataset = pd.merge(dataset, t18, on='item_price_level', how='left')

    t19 = feature[['item_price_level', 'is_trade']]
    t19['item_price_trade_total_times'] = t19['is_trade']
    t19 = t19.groupby('item_price_level')['item_price_trade_total_times'].agg('sum').reset_index()
    dataset = pd.merge(dataset, t19, on='item_price_level', how='left')
    dataset['price_transfer_rate'] = dataset['item_price_trade_total_times'] / dataset['item_price_click_total_times']

    t22 = feature[['item_collected_level']]
    t22['item_collected_click_total_times'] = 1
    t22 = t22.groupby('item_collected_level').agg('sum').reset_index()
    dataset = pd.merge(dataset, t22, on='item_collected_level', how='left')

    t23 = feature[['item_collected_level', 'is_trade']]
    t23['item_collected_trade_total_times'] = t23['is_trade']
    t23 = t23.groupby('item_collected_level')['item_collected_trade_total_times'].agg('sum').reset_index()
    dataset = pd.merge(dataset, t23, on='item_collected_level', how='left')
    dataset['collected_transfer_rate'] = dataset['item_collected_trade_total_times'] / dataset['item_collected_click_total_times']

    t24 = feature[['item_pv_level']]
    t24['item_pv_click_total_times'] = 1
    t24 = t24.groupby('item_pv_level').agg('sum').reset_index()
    dataset = pd.merge(dataset, t24, on='item_pv_level', how='left')

    t25 = feature[['item_pv_level', 'is_trade']]
    t25['item_pv_trade_total_times'] = t25['is_trade']
    t25 = t25.groupby('item_pv_level')['item_pv_trade_total_times'].agg('sum').reset_index()
    dataset = pd.merge(dataset, t25, on='item_pv_level', how='left')
    dataset['pv_transfer_rate'] = dataset['item_pv_trade_total_times'] / dataset['item_pv_click_total_times']

    t28 = feature[['shop_id']]
    t28['shop_is_click_total_times'] = 1
    t28 = t28.groupby('shop_id').agg('sum').reset_index()
    dataset = pd.merge(dataset, t28, on='shop_id', how='left')
    dataset.shop_is_click_total_times = dataset.shop_is_click_total_times.replace(np.nan, 0)

    t29 = feature[['shop_id', 'is_trade']]
    t29['shop_is_trade_times'] = t29['is_trade']
    t29 = t29.groupby('shop_id')['shop_is_trade_times'].agg('sum').reset_index()
    dataset = pd.merge(dataset, t29, on='shop_id', how='left')
    dataset.shop_is_trade_times = dataset.shop_is_trade_times.replace(np.nan, 0)

    # hyper3 = HyperParam(1, 1)
    # hyper3.update_from_data_by_FPI(dataset['shop_is_click_total_times'].values, dataset['shop_is_trade_times'].values,
    #                                1000,0.0001)
    # print('hyper3:', hyper3.alpha, hyper3.beta)
    alpha3 = 1.4222251297828883
    beta3 = 72.27949407200967

    # alpha3 = 1.3999236994298572
    # beta3 = 68.90493800983052

    # alpha3 = 1.4338266832602047
    # beta3 = 68.38019403696897

    # dataset['shop_quality'] = dataset['shop_is_trade_times'].apply(lambda x: float(x) + hyper3.alpha) / \
    #                           dataset['shop_is_click_total_times'].apply(lambda x: float(x) + hyper3.alpha + hyper3.beta)
    # dataset.shop_quality = dataset.shop_quality.replace(np.nan, hyper3.alpha / (hyper3.alpha + hyper3.beta))

    dataset['shop_quality'] = dataset['shop_is_trade_times'].apply(lambda x: float(x) + alpha3) / \
                              dataset['shop_is_click_total_times'].apply(lambda x: float(x) +alpha3 + beta3)
    dataset.shop_quality = dataset.shop_quality.replace(np.nan, alpha3 / (alpha3 + beta3))

    t32 = feature[['shop_star_level']]
    t32['shop_star_click_total_times'] = 1
    t32 = t32.groupby('shop_star_level').agg('sum').reset_index()
    dataset = pd.merge(dataset, t32, on='shop_star_level', how='left')

    t33 = feature[['shop_star_level', 'is_trade']]
    t33['shop_star_trade_total_times'] = t33['is_trade']
    t33 = t33.groupby('shop_star_level')['shop_star_trade_total_times'].agg('sum').reset_index()
    dataset = pd.merge(dataset, t33, on='shop_star_level', how='left')
    dataset['shop_star_transfer_rate'] = dataset['shop_star_trade_total_times'] / dataset[
        'shop_star_click_total_times']

    drop = ['user_click_total_times','user_buy_times','user_gender_click_times','user_gender_trade_times',
            'user_age_click_times','user_age_trade_times','user_star_click_times','user_star_trade_times',
            'item_is_click_total_times','item_is_trade_times','same_category_is_click_times',
            'same_category_is_buy_times','item_brand_click_total_times','item_brand_trade_total_times',
            'item_city_click_total_times','item_city_trade_total_times','item_price_click_total_times',
            'item_price_trade_total_times','item_collected_click_total_times','item_collected_trade_total_times',
            'item_pv_click_total_times','item_pv_trade_total_times','shop_is_click_total_times','shop_is_trade_times',
            'shop_star_click_total_times','shop_star_trade_total_times']

    dataset.drop(drop,axis=1,inplace=True)

    return dataset

# 交叉双特征
def get_two_feature(feature,label_feature):

    t=feature[['user_id','item_id']]
    t['user_click_same_item_times']=1
    t=t.groupby(['user_id','item_id']).agg('sum').reset_index()
    dataset = pd.merge(label_feature, t, on=['user_id','item_id'], how='left')

    t1=feature[['user_id','hour']]
    t1['user_hour_click_times']=1
    t1=t1.groupby(['user_id','hour']).agg('sum').reset_index()
    dataset = pd.merge(dataset, t1, on=['user_id', 'hour'], how='left')


    k = feature[['user_gender_id', 'last_category_id', 'is_trade']]
    k['gender_repeat_buy_same_category_times'] = k['is_trade']
    k=k.groupby(['user_gender_id', 'last_category_id'])['gender_repeat_buy_same_category_times'].agg('sum').reset_index()
    dataset=pd.merge(dataset,k,on=['user_gender_id', 'last_category_id'],how='left')

    k1 = feature[['user_gender_id', 'last_category_id']]
    k1['gender_repeat_click_same_category_times'] = 1
    k1 = k1.groupby(['user_gender_id', 'last_category_id']).agg('sum').reset_index()
    dataset = pd.merge(dataset, k1, on=['user_gender_id', 'last_category_id'], how='left')

    dataset['gender_category_transfer_rate'] = dataset['gender_repeat_buy_same_category_times']/dataset['gender_repeat_click_same_category_times']

    k2 = feature[['user_gender_id','item_id','is_trade']]
    k2['gender_repeat_buy_same_item_times'] = k2['is_trade']
    k2 = k2.groupby(['user_gender_id','item_id'])['gender_repeat_buy_same_item_times'].agg('sum').reset_index()
    dataset=pd.merge(dataset,k2,on=['user_gender_id','item_id'],how='left')

    k3 = feature[['user_gender_id', 'item_id']]
    k3['gender_repeat_click_same_item_times'] = 1
    k3 = k3.groupby(['user_gender_id', 'item_id']).agg('sum').reset_index()
    dataset = pd.merge(dataset, k3, on=['user_gender_id', 'item_id'], how='left')

    dataset['gender_item_transfer_rate'] = dataset['gender_repeat_buy_same_item_times'] / dataset['gender_repeat_click_same_item_times']

    k8 = feature[['user_gender_id', 'item_brand_id', 'is_trade']]
    k8['gender_repeat_buy_same_brand_times'] = k8['is_trade']
    k8 = k8.groupby(['user_gender_id', 'item_brand_id'])['gender_repeat_buy_same_brand_times'].agg('sum').reset_index()
    dataset = pd.merge(dataset, k8, on=['user_gender_id', 'item_brand_id'], how='left')

    k9 = feature[['user_gender_id', 'item_brand_id']]
    k9['gender_repeat_click_same_brand_times'] = 1
    k9 = k9.groupby(['user_gender_id', 'item_brand_id']).agg('sum').reset_index()
    dataset = pd.merge(dataset, k9, on=['user_gender_id', 'item_brand_id'], how='left')
    dataset['gender_brand_transfer_rate'] = dataset['gender_repeat_buy_same_brand_times']/dataset['gender_repeat_click_same_brand_times']

    k10 = feature[['user_gender_id', 'item_price_level']]
    k10['gender_repeat_click_same_price_times'] = 1
    k10 = k10.groupby(['user_gender_id', 'item_price_level']).agg('sum').reset_index()
    dataset = pd.merge(dataset, k10, on=['user_gender_id', 'item_price_level'], how='left')

    k11 = feature[['user_gender_id', 'item_price_level', 'is_trade']]
    k11['gender_repeat_buy_same_price_times'] = k11['is_trade']
    k11 = k11.groupby(['user_gender_id', 'item_price_level'])['gender_repeat_buy_same_price_times'].agg('sum').reset_index()
    dataset = pd.merge(dataset, k11, on=['user_gender_id', 'item_price_level'], how='left')

    dataset['gender_price_transfer_rate'] = dataset['gender_repeat_buy_same_price_times'] / dataset[
        'gender_repeat_click_same_price_times']


    occup3 = feature[['user_occupation_id', 'item_id', 'is_trade']]
    occup3['occupation_repeat_buy_same_item_times'] = occup3['is_trade']
    occup3 = occup3.groupby(['user_occupation_id', 'item_id'])['occupation_repeat_buy_same_item_times'].agg('sum').reset_index()
    dataset = pd.merge(dataset, occup3, on=['user_occupation_id', 'item_id'], how='left')

    occup4 = feature[['user_occupation_id', 'item_id']]
    occup4['occupation_repeat_click_same_item_times'] = 1
    occup4 = occup4.groupby(['user_occupation_id', 'item_id']).agg('sum').reset_index()
    dataset = pd.merge(dataset, occup4, on=['user_occupation_id', 'item_id'], how='left')

    dataset['occupation_item_transfer_rate'] = dataset['occupation_repeat_buy_same_item_times'] / \
                                                   dataset['occupation_repeat_click_same_item_times']

    occup9 = feature[['user_occupation_id', 'item_brand_id', 'is_trade']]
    occup9['occupation_repeat_buy_same_brand_times'] = occup9['is_trade']
    occup9 = occup9.groupby(['user_occupation_id', 'item_brand_id'])['occupation_repeat_buy_same_brand_times'].agg('sum').reset_index()
    dataset = pd.merge(dataset, occup9, on=['user_occupation_id', 'item_brand_id'], how='left')

    occup10 = feature[['user_occupation_id', 'item_brand_id']]
    occup10['occupation_repeat_click_same_brand_times'] = 1
    occup10 = occup10.groupby(['user_occupation_id', 'item_brand_id']).agg('sum').reset_index()
    dataset = pd.merge(dataset, occup10, on=['user_occupation_id', 'item_brand_id'], how='left')
    dataset['occupation_brand_transfer_rate'] = dataset['occupation_repeat_buy_same_brand_times']/ \
                                               dataset['occupation_repeat_click_same_brand_times']

    occup11 = feature[['user_occupation_id', 'item_price_level', 'is_trade']]
    occup11['occupation_repeat_buy_same_price_times'] = occup11['is_trade']
    occup11 = occup11.groupby(['user_occupation_id', 'item_price_level'])['occupation_repeat_buy_same_price_times'].agg(
        'sum').reset_index()
    dataset = pd.merge(dataset, occup11, on=['user_occupation_id', 'item_price_level'], how='left')

    occup12 = feature[['user_occupation_id', 'item_price_level']]
    occup12['occupation_repeat_click_same_price_times'] = 1
    occup12 = occup12.groupby(['user_occupation_id', 'item_price_level']).agg('sum').reset_index()
    dataset = pd.merge(dataset, occup12, on=['user_occupation_id', 'item_price_level'], how='left')
    dataset['occupation_price_transfer_rate'] = dataset['occupation_repeat_buy_same_price_times'] / \
                                                dataset['occupation_repeat_click_same_price_times']

    occup23 = feature[['user_occupation_id','hour']]
    occup23['occupation_hour_click_times'] =1
    occup23 = occup23.groupby(['user_occupation_id','hour']).agg('sum').reset_index()
    dataset = pd.merge(dataset,occup23,on=['user_occupation_id','hour'],how='left')

    occup24 = feature[['user_occupation_id', 'hour','is_trade']]
    occup24['occupation_hour_buy_times'] = occup24['is_trade']
    occup24 = occup24.groupby(['user_occupation_id', 'hour'])['occupation_hour_buy_times'].agg('sum').reset_index()
    dataset = pd.merge(dataset, occup24, on=['user_occupation_id', 'hour'], how='left')

    dataset['occupation_hour_transfer_rate'] = dataset['occupation_hour_buy_times'] /  dataset['occupation_hour_click_times']

    age1 = feature[['user_age_level', 'last_category_id', 'is_trade']]
    age1['age_repeat_buy_same_category_times'] = age1['is_trade']
    age1 = age1.groupby(['user_age_level', 'last_category_id'])['age_repeat_buy_same_category_times'].agg('sum').reset_index()
    dataset = pd.merge(dataset, age1, on=['user_age_level', 'last_category_id'], how='left')

    age2 = feature[['user_age_level', 'last_category_id']]
    age2['age_repeat_click_same_category_times'] = 1
    age2 = age2.groupby(['user_age_level', 'last_category_id']).agg('sum').reset_index()
    dataset = pd.merge(dataset, age2, on=['user_age_level', 'last_category_id'], how='left')
    dataset['age_category_transfer_rate'] = dataset['age_repeat_buy_same_category_times'] / \
                                        dataset['age_repeat_click_same_category_times']

    age3 = feature[['user_age_level', 'item_id', 'is_trade']]
    age3['age_repeat_buy_same_item_times'] = age3['is_trade']
    age3 = age3.groupby(['user_age_level', 'item_id'])['age_repeat_buy_same_item_times'].agg('sum').reset_index()
    dataset = pd.merge(dataset, age3, on=['user_age_level', 'item_id'], how='left')

    age4 = feature[['user_age_level', 'item_id']]
    age4['age_repeat_click_same_item_times'] = 1
    age4 = age4.groupby(['user_age_level', 'item_id']).agg('sum').reset_index()
    dataset = pd.merge(dataset, age4, on=['user_age_level', 'item_id'], how='left')
    dataset['age_item_transfer_rate'] = dataset['age_repeat_buy_same_item_times'] / \
                                            dataset['age_repeat_click_same_item_times']

    age9 = feature[['user_age_level', 'item_brand_id', 'is_trade']]
    age9['age_repeat_buy_same_brand_times'] = age9['is_trade']
    age9 = age9.groupby(['user_age_level', 'item_brand_id'])['age_repeat_buy_same_brand_times'].agg('sum').reset_index()
    dataset = pd.merge(dataset, age9, on=['user_age_level', 'item_brand_id'], how='left')

    age10 = feature[['user_age_level', 'item_brand_id']]
    age10['age_repeat_click_same_brand_times'] = 1
    age10 = age10.groupby(['user_age_level', 'item_brand_id']).agg('sum').reset_index()
    dataset = pd.merge(dataset, age10, on=['user_age_level', 'item_brand_id'], how='left')
    dataset['age_brand_transfer_rate'] = dataset['age_repeat_buy_same_brand_times'] / \
                                        dataset['age_repeat_click_same_brand_times']

    age11 = feature[['user_age_level', 'item_price_level', 'is_trade']]
    age11['age_repeat_buy_same_price_times'] = age11['is_trade']
    age11 = age11.groupby(['user_age_level', 'item_price_level'])['age_repeat_buy_same_price_times'].agg('sum').reset_index()
    dataset = pd.merge(dataset, age11, on=['user_age_level', 'item_price_level'], how='left')

    age12 = feature[['user_age_level', 'item_price_level']]
    age12['age_repeat_click_same_price_times'] = 1
    age12 = age12.groupby(['user_age_level', 'item_price_level']).agg('sum').reset_index()
    dataset = pd.merge(dataset, age12, on=['user_age_level', 'item_price_level'], how='left')
    dataset['age_price_transfer_rate'] = dataset['age_repeat_buy_same_price_times'] / \
                                         dataset['age_repeat_click_same_price_times']

    star9 = feature[['user_star_level', 'item_brand_id', 'is_trade']]
    star9['star_repeat_buy_same_brand_times'] = star9['is_trade']
    star9 = star9.groupby(['user_star_level', 'item_brand_id'])['star_repeat_buy_same_brand_times'].agg('sum').reset_index()
    dataset = pd.merge(dataset, star9, on=['user_star_level', 'item_brand_id'], how='left')

    star10 = feature[['user_star_level', 'item_brand_id']]
    star10['star_repeat_click_same_brand_times'] = 1
    star10 = star10.groupby(['user_star_level', 'item_brand_id']).agg('sum').reset_index()
    dataset = pd.merge(dataset, star10, on=['user_star_level', 'item_brand_id'], how='left')

    dataset['star_brand_transfer_rate'] = dataset['star_repeat_buy_same_brand_times'] / \
                                         dataset['star_repeat_click_same_brand_times']

    star11 = feature[['user_star_level', 'item_price_level', 'is_trade']]
    star11['star_repeat_buy_same_price_times'] = star11['is_trade']
    star11 = star11.groupby(['user_star_level', 'item_price_level'])['star_repeat_buy_same_price_times'].agg(
        'sum').reset_index()
    dataset = pd.merge(dataset, star11, on=['user_star_level', 'item_price_level'], how='left')

    star12 = feature[['user_star_level', 'item_price_level']]
    star12['star_repeat_click_same_price_times'] = 1
    star12 = star12.groupby(['user_star_level', 'item_price_level']).agg('sum').reset_index()
    dataset = pd.merge(dataset, star12, on=['user_star_level', 'item_price_level'], how='left')

    dataset['star_price_transfer_rate'] = dataset['star_repeat_buy_same_price_times'] / \
                                          dataset['star_repeat_click_same_price_times']

    star19 = feature[['user_star_level', 'shop_review_num_level', 'is_trade']]
    star19['star_repeat_buy_review_num_times'] = star19['is_trade']
    star19 = star19.groupby(['user_star_level', 'shop_review_num_level'])['star_repeat_buy_review_num_times'].agg(
        'sum').reset_index()
    dataset = pd.merge(dataset, star19, on=['user_star_level', 'shop_review_num_level'], how='left')

    star20 = feature[['user_star_level', 'shop_review_num_level']]
    star20['star_repeat_click_review_num_times'] = 1
    star20 = star20.groupby(['user_star_level', 'shop_review_num_level']).agg('sum').reset_index()
    dataset = pd.merge(dataset, star20, on=['user_star_level', 'shop_review_num_level'], how='left')

    dataset['star_review_num_transfer_rate'] = dataset['star_repeat_buy_review_num_times'] / \
                                              dataset['star_repeat_click_review_num_times']

    star21 = feature[['user_star_level', 'shop_star_level', 'is_trade']]
    star21['star_repeat_buy_star_times'] = star21['is_trade']
    star21 = star21.groupby(['user_star_level', 'shop_star_level'])['star_repeat_buy_star_times'].agg(
        'sum').reset_index()
    dataset = pd.merge(dataset, star21, on=['user_star_level', 'shop_star_level'], how='left')

    star22 = feature[['user_star_level', 'shop_star_level']]
    star22['star_repeat_click_star_times'] = 1
    star22 = star22.groupby(['user_star_level', 'shop_star_level']).agg('sum').reset_index()
    dataset = pd.merge(dataset, star22, on=['user_star_level', 'shop_star_level'], how='left')

    dataset['star_star_transfer_rate'] = dataset['star_repeat_buy_star_times'] / \
                                               dataset['star_repeat_click_star_times']

    drop = ['gender_repeat_buy_same_category_times','gender_repeat_click_same_category_times',
            'gender_repeat_buy_same_item_times','gender_repeat_click_same_item_times',
            'gender_repeat_buy_same_brand_times','gender_repeat_click_same_brand_times',
            'gender_repeat_click_same_price_times','gender_repeat_buy_same_price_times',

            'occupation_repeat_buy_same_item_times', 'occupation_repeat_click_same_item_times',
            'occupation_repeat_buy_same_brand_times', 'occupation_repeat_click_same_brand_times',
            'occupation_repeat_click_same_price_times', 'occupation_repeat_buy_same_price_times',

            'age_repeat_buy_same_category_times', 'age_repeat_click_same_category_times',
            'age_repeat_buy_same_item_times', 'age_repeat_click_same_item_times',
            'age_repeat_buy_same_brand_times', 'age_repeat_click_same_brand_times',
            'age_repeat_click_same_price_times', 'age_repeat_buy_same_price_times',

            'star_repeat_buy_same_brand_times', 'star_repeat_click_same_brand_times',
            'star_repeat_click_same_price_times', 'star_repeat_buy_same_price_times',
            'star_repeat_buy_review_num_times','star_repeat_click_review_num_times',
            'star_repeat_buy_star_times','star_repeat_click_star_times']

    dataset.drop(drop,axis=1,inplace=True)
    return dataset

# 交叉三特征
def get_three_feature(feature,dataset):

    t=feature[['user_gender_id','user_age_level','item_brand_id']]
    t['gender_age_click_brand_times']=1
    t=t.groupby(['user_gender_id','user_age_level','item_brand_id']).agg('sum').reset_index()
    dataset=pd.merge(dataset,t,on=['user_gender_id','user_age_level','item_brand_id'],how='left')

    t1 = feature[['user_gender_id', 'user_age_level', 'item_brand_id','is_trade']]
    t1['gender_age_buy_brand_times'] = t1['is_trade']
    t1 = t1.groupby(['user_gender_id', 'user_age_level', 'item_brand_id'])['gender_age_buy_brand_times'].\
        agg('sum').reset_index()
    dataset=pd.merge(dataset,t1,on=['user_gender_id','user_age_level','item_brand_id'],how='left')

    dataset['gender_age_brand_transfer_rate'] = dataset['gender_age_buy_brand_times'] / dataset['gender_age_click_brand_times']

    t2=feature[['user_gender_id','user_age_level','item_category_list1']]
    t2['gender_age_click_category_times'] = 1
    t2 = t2.groupby(['user_gender_id', 'user_age_level', 'item_category_list1']).agg('sum').reset_index()
    dataset = pd.merge(dataset, t2, on=['user_gender_id', 'user_age_level', 'item_category_list1'], how='left')

    t3 = feature[['user_gender_id', 'user_age_level', 'item_category_list1', 'is_trade']]
    t3['gender_age_buy_category_times'] = t3['is_trade']
    t3 = t3.groupby(['user_gender_id', 'user_age_level', 'item_category_list1'])['gender_age_buy_category_times']. \
        agg('sum').reset_index()
    dataset = pd.merge(dataset, t3, on=['user_gender_id', 'user_age_level', 'item_category_list1'], how='left')
    dataset['gender_age_category_transfer_rate'] = dataset['gender_age_buy_category_times'] / dataset[
        'gender_age_click_category_times']

    t6 = feature[['user_gender_id', 'user_age_level', 'item_id']]
    t6['gender_age_click_item_times'] = 1
    t6 = t6.groupby(['user_gender_id', 'user_age_level', 'item_id']).agg('sum').reset_index()
    dataset = pd.merge(dataset, t6, on=['user_gender_id', 'user_age_level', 'item_id'], how='left')

    t7 = feature[['user_gender_id', 'user_age_level', 'item_id', 'is_trade']]
    t7['gender_age_buy_item_times'] = t7['is_trade']
    t7 = t7.groupby(['user_gender_id', 'user_age_level', 'item_id'])['gender_age_buy_item_times']. \
        agg('sum').reset_index()
    dataset = pd.merge(dataset, t7, on=['user_gender_id', 'user_age_level', 'item_id'], how='left')
    dataset['gender_age_item_transfer_rate'] = dataset['gender_age_buy_item_times'] / dataset[
        'gender_age_click_item_times']

    t10 = feature[['user_gender_id', 'user_age_level', 'item_price_level']]
    t10['gender_age_click_price_times'] = 1
    t10 = t10.groupby(['user_gender_id', 'user_age_level', 'item_price_level']).agg('sum').reset_index()
    dataset = pd.merge(dataset, t10, on=['user_gender_id', 'user_age_level', 'item_price_level'], how='left')

    t11 = feature[['user_gender_id', 'user_age_level', 'item_price_level', 'is_trade']]
    t11['gender_age_buy_price_times'] = t11['is_trade']
    t11 = t11.groupby(['user_gender_id', 'user_age_level', 'item_price_level'])['gender_age_buy_price_times']. \
        agg('sum').reset_index()
    dataset = pd.merge(dataset, t11, on=['user_gender_id', 'user_age_level', 'item_price_level'], how='left')
    dataset['gender_age_price_transfer_rate'] = dataset['gender_age_buy_price_times'] / dataset[
        'gender_age_click_price_times']

    go6 = feature[['user_gender_id', 'user_occupation_id', 'item_id']]
    go6['gender_occupation_click_item_times'] = 1
    go6 = go6.groupby(['user_gender_id', 'user_occupation_id', 'item_id']).agg('sum').reset_index()
    dataset = pd.merge(dataset, go6, on=['user_gender_id', 'user_occupation_id', 'item_id'], how='left')

    go7 = feature[['user_gender_id', 'user_occupation_id', 'item_id', 'is_trade']]
    go7['gender_occupation_buy_item_times'] = go7['is_trade']
    go7 = go7.groupby(['user_gender_id', 'user_occupation_id', 'item_id'])['gender_occupation_buy_item_times']. \
        agg('sum').reset_index()
    dataset = pd.merge(dataset, go7, on=['user_gender_id', 'user_occupation_id', 'item_id'], how='left')
    dataset['gender_occupation_item_transfer_rate'] = dataset['gender_occupation_buy_item_times'] / dataset[
        'gender_occupation_click_item_times']

    go10 = feature[['user_gender_id', 'user_occupation_id', 'item_price_level']]
    go10['gender_occupation_click_price_times'] = 1
    go10 = go10.groupby(['user_gender_id', 'user_occupation_id', 'item_price_level']).agg('sum').reset_index()
    dataset = pd.merge(dataset, go10, on=['user_gender_id', 'user_occupation_id', 'item_price_level'], how='left')

    go11 = feature[['user_gender_id', 'user_occupation_id', 'item_price_level', 'is_trade']]
    go11['gender_occupation_buy_price_times'] = go11['is_trade']
    go11 = go11.groupby(['user_gender_id', 'user_occupation_id', 'item_price_level'])['gender_occupation_buy_price_times']. \
        agg('sum').reset_index()
    dataset = pd.merge(dataset, go11, on=['user_gender_id', 'user_occupation_id', 'item_price_level'], how='left')
    dataset['gender_occupation_price_transfer_rate'] = dataset['gender_occupation_buy_price_times'] / dataset[
        'gender_occupation_click_price_times']

    gs = feature[['user_gender_id', 'user_star_level', 'item_brand_id']]
    gs['gender_star_click_brand_times'] = 1
    gs = gs.groupby(['user_gender_id', 'user_star_level', 'item_brand_id']).agg('sum').reset_index()
    dataset = pd.merge(dataset, gs, on=['user_gender_id', 'user_star_level', 'item_brand_id'], how='left')

    gs1 = feature[['user_gender_id', 'user_star_level', 'item_brand_id', 'is_trade']]
    gs1['gender_star_buy_brand_times'] = gs1['is_trade']
    gs1 = gs1.groupby(['user_gender_id', 'user_star_level', 'item_brand_id'])['gender_star_buy_brand_times']. \
        agg('sum').reset_index()
    dataset = pd.merge(dataset, gs1, on=['user_gender_id', 'user_star_level', 'item_brand_id'], how='left')

    dataset['gender_star_brand_transfer_rate'] = dataset['gender_star_buy_brand_times'] / dataset[
        'gender_star_click_brand_times']
    
    gs6 = feature[['user_gender_id', 'user_star_level', 'item_id']]
    gs6['gender_star_click_item_times'] = 1
    gs6 = gs6.groupby(['user_gender_id', 'user_star_level', 'item_id']).agg('sum').reset_index()
    dataset = pd.merge(dataset, gs6, on=['user_gender_id', 'user_star_level', 'item_id'], how='left')

    gs7 = feature[['user_gender_id', 'user_star_level', 'item_id', 'is_trade']]
    gs7['gender_star_buy_item_times'] = gs7['is_trade']
    gs7 = gs7.groupby(['user_gender_id', 'user_star_level', 'item_id'])['gender_star_buy_item_times']. \
        agg('sum').reset_index()
    dataset = pd.merge(dataset, gs7, on=['user_gender_id', 'user_star_level', 'item_id'], how='left')
    dataset['gender_star_item_transfer_rate'] = dataset['gender_star_buy_item_times'] / dataset[
        'gender_star_click_item_times']

    gs10 = feature[['user_gender_id', 'user_star_level', 'item_price_level']]
    gs10['gender_star_click_price_times'] = 1
    gs10 = gs10.groupby(['user_gender_id', 'user_star_level', 'item_price_level']).agg('sum').reset_index()
    dataset = pd.merge(dataset, gs10, on=['user_gender_id', 'user_star_level', 'item_price_level'], how='left')

    gs11 = feature[['user_gender_id', 'user_star_level', 'item_price_level', 'is_trade']]
    gs11['gender_star_buy_price_times'] = gs11['is_trade']
    gs11 = gs11.groupby(['user_gender_id', 'user_star_level', 'item_price_level'])[
        'gender_star_buy_price_times']. \
        agg('sum').reset_index()
    dataset = pd.merge(dataset, gs11, on=['user_gender_id', 'user_star_level', 'item_price_level'], how='left')
    dataset['gender_star_price_transfer_rate'] = dataset['gender_star_buy_price_times'] / dataset[
        'gender_star_click_price_times']

    
    ags = feature[['user_age_level', 'user_star_level', 'item_brand_id']]
    ags['age_star_click_brand_times'] = 1
    ags = ags.groupby(['user_age_level', 'user_star_level', 'item_brand_id']).agg('sum').reset_index()
    dataset = pd.merge(dataset, ags, on=['user_age_level', 'user_star_level', 'item_brand_id'], how='left')

    ags1 = feature[['user_age_level', 'user_star_level', 'item_brand_id', 'is_trade']]
    ags1['age_star_buy_brand_times'] = ags1['is_trade']
    ags1 = ags1.groupby(['user_age_level', 'user_star_level', 'item_brand_id'])['age_star_buy_brand_times']. \
        agg('sum').reset_index()
    dataset = pd.merge(dataset, ags1, on=['user_age_level', 'user_star_level', 'item_brand_id'], how='left')

    dataset['age_star_brand_transfer_rate'] = dataset['age_star_buy_brand_times'] / dataset[
        'age_star_click_brand_times']

    ags2 = feature[['user_age_level', 'user_star_level', 'item_category_list1']]
    ags2['age_star_click_category_times'] = 1
    ags2 = ags2.groupby(['user_age_level', 'user_star_level', 'item_category_list1']).agg('sum').reset_index()
    dataset = pd.merge(dataset, ags2, on=['user_age_level', 'user_star_level', 'item_category_list1'], how='left')

    ags3 = feature[['user_age_level', 'user_star_level', 'item_category_list1', 'is_trade']]
    ags3['age_star_buy_category_times'] = ags3['is_trade']
    ags3 = ags3.groupby(['user_age_level', 'user_star_level', 'item_category_list1'])[
        'age_star_buy_category_times']. \
        agg('sum').reset_index()
    dataset = pd.merge(dataset, ags3, on=['user_age_level', 'user_star_level', 'item_category_list1'], how='left')
    dataset['age_star_category_transfer_rate'] = dataset['age_star_buy_category_times'] / dataset[
        'age_star_click_category_times']

    ags6 = feature[['user_age_level', 'user_star_level', 'item_id']]
    ags6['age_star_click_item_times'] = 1
    ags6 = ags6.groupby(['user_age_level', 'user_star_level', 'item_id']).agg('sum').reset_index()
    dataset = pd.merge(dataset, ags6, on=['user_age_level', 'user_star_level', 'item_id'], how='left')

    ags7 = feature[['user_age_level', 'user_star_level', 'item_id', 'is_trade']]
    ags7['age_star_buy_item_times'] = ags7['is_trade']
    ags7 = ags7.groupby(['user_age_level', 'user_star_level', 'item_id'])['age_star_buy_item_times']. \
        agg('sum').reset_index()
    dataset = pd.merge(dataset, ags7, on=['user_age_level', 'user_star_level', 'item_id'], how='left')
    dataset['age_star_item_transfer_rate'] = dataset['age_star_buy_item_times'] / dataset[
        'age_star_click_item_times']

    ags10 = feature[['user_age_level', 'user_star_level', 'item_price_level']]
    ags10['age_star_click_price_times'] = 1
    ags10 = ags10.groupby(['user_age_level', 'user_star_level', 'item_price_level']).agg('sum').reset_index()
    dataset = pd.merge(dataset, ags10, on=['user_age_level', 'user_star_level', 'item_price_level'], how='left')

    ags11 = feature[['user_age_level', 'user_star_level', 'item_price_level', 'is_trade']]
    ags11['age_star_buy_price_times'] = ags11['is_trade']
    ags11 = ags11.groupby(['user_age_level', 'user_star_level', 'item_price_level'])[
        'age_star_buy_price_times']. \
        agg('sum').reset_index()
    dataset = pd.merge(dataset, ags11, on=['user_age_level', 'user_star_level', 'item_price_level'], how='left')
    dataset['age_star_price_transfer_rate'] = dataset['age_star_buy_price_times'] / dataset[
        'age_star_click_price_times']


    ago6 = feature[['user_age_level', 'user_occupation_id', 'item_id']]
    ago6['age_occupation_click_item_times'] = 1
    ago6 = ago6.groupby(['user_age_level', 'user_occupation_id', 'item_id']).agg('sum').reset_index()
    dataset = pd.merge(dataset, ago6, on=['user_age_level', 'user_occupation_id', 'item_id'], how='left')

    ago7 = feature[['user_age_level', 'user_occupation_id', 'item_id', 'is_trade']]
    ago7['age_occupation_buy_item_times'] = ago7['is_trade']
    ago7 = ago7.groupby(['user_age_level', 'user_occupation_id', 'item_id'])['age_occupation_buy_item_times']. \
        agg('sum').reset_index()
    dataset = pd.merge(dataset, ago7, on=['user_age_level', 'user_occupation_id', 'item_id'], how='left')
    dataset['age_occupation_item_transfer_rate'] = dataset['age_occupation_buy_item_times'] / dataset[
        'age_occupation_click_item_times']
    
    ago10 = feature[['user_age_level', 'user_occupation_id', 'item_price_level']]
    ago10['age_occupation_click_price_times'] = 1
    ago10 = ago10.groupby(['user_age_level', 'user_occupation_id', 'item_price_level']).agg('sum').reset_index()
    dataset = pd.merge(dataset, ago10, on=['user_age_level', 'user_occupation_id', 'item_price_level'], how='left')

    ago11 = feature[['user_age_level', 'user_occupation_id', 'item_price_level', 'is_trade']]
    ago11['age_occupation_buy_price_times'] = ago11['is_trade']
    ago11 = ago11.groupby(['user_age_level', 'user_occupation_id', 'item_price_level'])[
        'age_occupation_buy_price_times']. \
        agg('sum').reset_index()
    dataset = pd.merge(dataset, ago11, on=['user_age_level', 'user_occupation_id', 'item_price_level'], how='left')
    dataset['age_occupation_price_transfer_rate'] = dataset['age_occupation_buy_price_times'] / dataset[
        'age_occupation_click_price_times']

    
    sto10 = feature[['user_star_level', 'user_occupation_id', 'item_price_level']]
    sto10['star_occupation_click_price_times'] = 1
    sto10 = sto10.groupby(['user_star_level', 'user_occupation_id', 'item_price_level']).agg('sum').reset_index()
    dataset = pd.merge(dataset, sto10, on=['user_star_level', 'user_occupation_id', 'item_price_level'], how='left')

    sto11 = feature[['user_star_level', 'user_occupation_id', 'item_price_level', 'is_trade']]
    sto11['star_occupation_buy_price_times'] = sto11['is_trade']
    sto11 = sto11.groupby(['user_star_level', 'user_occupation_id', 'item_price_level'])[
        'star_occupation_buy_price_times']. \
        agg('sum').reset_index()
    dataset = pd.merge(dataset, sto11, on=['user_star_level', 'user_occupation_id', 'item_price_level'], how='left')
    dataset['star_occupation_price_transfer_rate'] = dataset['star_occupation_buy_price_times'] / dataset[
        'star_occupation_click_price_times']

    
    drop = ['gender_age_click_brand_times','gender_age_buy_brand_times','gender_age_buy_category_times',
            'gender_age_click_category_times','gender_age_buy_item_times','gender_age_click_item_times',
            'gender_age_buy_price_times','gender_age_click_price_times',

            'gender_occupation_buy_item_times', 'gender_occupation_click_item_times',
            'gender_occupation_buy_price_times', 'gender_occupation_click_price_times',

            'gender_star_click_brand_times', 'gender_star_buy_brand_times',
            'gender_star_buy_item_times', 'gender_star_click_item_times',
            'gender_star_buy_price_times', 'gender_star_click_price_times',

            'age_star_click_brand_times', 'age_star_buy_brand_times', 'age_star_buy_category_times',
            'age_star_click_category_times','age_star_buy_price_times', 'age_star_click_price_times',
            'age_occupation_buy_item_times', 'age_occupation_click_item_times',
            'star_occupation_buy_price_times', 'star_occupation_click_price_times']


    dataset.drop(drop,axis=1,inplace=True)
    return dataset

# 统计平均特征
def get_avg_feature(feature,dataset):

    t=feature[['item_id','day']]
    t['item_day_click_times']=1
    t=t.groupby(['item_id','day']).agg('sum').reset_index()
    t['item_day_avg_click_times']=t['item_day_click_times']
    t=t.groupby('item_id')['item_day_avg_click_times'].agg(lambda x:sum(list(x))/len(list(x))).reset_index()
    dataset=pd.merge(dataset,t,on='item_id',how='left')


    t1=feature[['item_id','day','is_trade']]
    t1['item_day_trade_times'] = t1['is_trade']
    t1 = t1.groupby(['item_id','day'])['item_day_trade_times'].agg(lambda x:sum(x==1)).reset_index()
    t1['item_day_avg_trade_times'] = t1['item_day_trade_times']
    t1 = t1.groupby('item_id')['item_day_avg_trade_times'].agg(lambda x: sum(list(x)) / len(list(x))).reset_index()
    dataset = pd.merge(dataset, t1, on='item_id', how='left')

    dataset['item_day_avg_transfer_rate']=dataset['item_day_avg_trade_times'] / dataset['item_day_avg_click_times']

    t2=feature[['user_id','day']]
    t2['user_day_avg_click_times'] = 1
    t2=t2.groupby(['user_id','day']).agg('sum').reset_index()
    t2=t2.groupby('user_id')['user_day_avg_click_times'].agg(lambda x: sum(list(x)) / len(list(x))).reset_index()
    dataset=pd.merge(dataset,t2,on='user_id',how='left')


    t3 = feature[['user_id', 'day','is_trade']]
    t3['user_day_avg_trade_times'] = t3['is_trade']
    t3 = t3.groupby(['user_id', 'day'])['user_day_avg_trade_times'].agg(lambda x:sum(x==1)).reset_index()
    t3 = t3.groupby('user_id')['user_day_avg_trade_times'].agg(lambda x: sum(list(x)) / len(list(x))).reset_index()
    dataset = pd.merge(dataset, t3, on='user_id', how='left')


    dataset['user_day_transfer_rate'] = dataset['user_day_avg_trade_times'] / dataset['user_day_avg_click_times']


    t4=feature[['last_category_id','day','is_trade']]
    t4['category_day_avg_trade_times']=t4['is_trade']
    t4=t4.groupby(['last_category_id','day'])['category_day_avg_trade_times'].agg(lambda x:sum(x==1)).reset_index()
    t4=t4.groupby('last_category_id')['category_day_avg_trade_times'].agg(lambda x: sum(list(x)) / len(list(x))).reset_index()
    dataset=pd.merge(dataset,t4,on='last_category_id',how='left')

    t5=feature[['item_brand_id','day']]
    t5['brand_day_avg_click_times']=1
    t5=t5.groupby(['item_brand_id','day']).agg('sum').reset_index()
    t5=t5.groupby('item_brand_id')['brand_day_avg_click_times'].agg(lambda x: sum(list(x)) / len(list(x))).reset_index()
    dataset=pd.merge(dataset,t5,on='item_brand_id',how='left')


    t6=feature[['item_brand_id','day','is_trade']]
    t6['brand_day_avg_trade_times']=t6['is_trade']
    t6=t6.groupby(['item_brand_id','day'])['brand_day_avg_trade_times'].agg(lambda x:sum(x==1)).reset_index()
    t6=t6.groupby('item_brand_id')['brand_day_avg_trade_times'].agg(lambda x: sum(list(x)) / len(list(x))).reset_index()
    dataset = pd.merge(dataset, t6, on='item_brand_id', how='left')


    dataset['brand_day_transfer_rate'] = dataset['brand_day_avg_trade_times'] / dataset['brand_day_avg_click_times']


    t7=feature[['item_city_id','day']]
    t7['city_day_avg_click_times']=1
    t7=t7.groupby(['item_city_id','day']).agg('sum').reset_index()
    t7=t7.groupby('item_city_id')['city_day_avg_click_times'].agg(lambda x: sum(list(x)) / len(list(x))).reset_index()
    dataset=pd.merge(dataset,t7,on='item_city_id',how='left')


    t8=feature[['item_city_id','day','is_trade']]
    t8['city_day_avg_trade_times']=t8['is_trade']
    t8=t8.groupby(['item_city_id','day'])['city_day_avg_trade_times'].agg(lambda x:sum(x==1)).reset_index()
    t8=t8.groupby('item_city_id')['city_day_avg_trade_times'].agg(lambda x: sum(list(x)) / len(list(x))).reset_index()
    dataset=pd.merge(dataset,t8,on='item_city_id',how='left')

    dataset['city_day_transfer_rate'] = dataset['city_day_avg_trade_times'] / dataset['city_day_avg_click_times']

    t9=feature[['hour','is_trade']]
    t9['hour_trade_times']=t9['is_trade']
    t9=t9.groupby('hour')['hour_trade_times'].agg('sum').reset_index()
    dataset=pd.merge(dataset,t9,on='hour',how='left')

    return dataset

# 时间差特征(用户点击时间差:标记第一次点击和最后一次点击)
def doTrick(data):

    subset = ['user_id','item_id', 'shop_id']
    data['maybe'] = 0
    pos = data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'maybe'] = 1
    pos = (~data.duplicated(subset=subset, keep='first')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'maybe'] = 2
    pos = (~data.duplicated(subset=subset, keep='last')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'maybe'] = 3

    # features_trans = ['maybe']
    # data = pd.get_dummies(data, columns=features_trans)
    # data['maybe_0'] = data['maybe_0'].astype(np.int8)
    # data['maybe_1'] = data['maybe_1'].astype(np.int8)
    # data['maybe_2'] = data['maybe_2'].astype(np.int8)
    # data['maybe_3'] = data['maybe_3'].astype(np.int8)

    # 时间差Trick
    temp = data.loc[:,['context_timestamp', 'user_id','item_id','shop_id']].drop_duplicates(subset=subset, keep='first')
    temp.rename(columns={'context_timestamp': 'diffTime_first'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['diffTime_first'] = data['context_timestamp'] - data['diffTime_first']
    del temp,pos

    temp = data.loc[:,['context_timestamp', 'user_id','item_id','shop_id']].drop_duplicates(subset=subset, keep='last')
    temp.rename(columns={'context_timestamp': 'diffTime_last'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['diffTime_last'] = data['diffTime_last'] - data['context_timestamp']
    del temp

    data.loc[~data.duplicated(subset=subset, keep=False), ['diffTime_first', 'diffTime_last']] = -1 #置0会变差
    return data

def doTrick2(data):

    subset = ['user_id']
    data['umaybe'] = 0
    pos = data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'umaybe'] = 1
    pos = (~data.duplicated(subset=subset, keep='first')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'umaybe'] = 2
    pos = (~data.duplicated(subset=subset, keep='last')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'umaybe'] = 3
    del pos

    # features_trans = ['umaybe']
    # data = pd.get_dummies(data, columns=features_trans)
    # data['umaybe_0'] = data['umaybe_0'].astype(np.int8)
    # data['umaybe_1'] = data['umaybe_1'].astype(np.int8)
    # data['umaybe_2'] = data['umaybe_2'].astype(np.int8)
    # data['umaybe_3'] = data['umaybe_3'].astype(np.int8)

    temp = data[['context_timestamp','user_id']]
    temp = temp.drop_duplicates(subset=subset, keep='first')
    temp.rename(columns={'context_timestamp': 'udiffTime_first'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['udiffTime_first'] = data['context_timestamp'] - data['udiffTime_first']
    del temp

    temp = data[['context_timestamp','user_id']]
    temp = temp.drop_duplicates(subset=subset, keep='last')
    temp.rename(columns={'context_timestamp': 'udiffTime_last'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['udiffTime_last'] = data['udiffTime_last'] - data['context_timestamp']
    del temp

    data.loc[~data.duplicated(subset=subset, keep=False), ['udiffTime_first', 'udiffTime_last']] = -1


    return data

def do_brand_trick(data):

    subset = ['user_id','item_brand_id']
    data['bmaybe'] = 0
    pos = data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'bmaybe'] = 1
    pos = (~data.duplicated(subset=subset, keep='first')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'bmaybe'] = 2
    pos = (~data.duplicated(subset=subset, keep='last')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'bmaybe'] = 3
    del pos

    # features_trans = ['bmaybe']
    # data = pd.get_dummies(data, columns=features_trans)
    # data['bmaybe_0'] = data['bmaybe_0'].astype(np.int8)
    # data['bmaybe_1'] = data['bmaybe_1'].astype(np.int8)
    # data['bmaybe_2'] = data['bmaybe_2'].astype(np.int8)
    # data['bmaybe_3'] = data['bmaybe_3'].astype(np.int8)

    temp = data[['context_timestamp','user_id','item_brand_id']]
    temp = temp.drop_duplicates(subset=subset, keep='first')
    temp.rename(columns={'context_timestamp': 'bdiffTime_first'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['bdiffTime_first'] = data['context_timestamp'] - data['bdiffTime_first']
    del temp

    temp = data[['context_timestamp','user_id','item_brand_id']]
    temp = temp.drop_duplicates(subset=subset, keep='last')
    temp.rename(columns={'context_timestamp': 'bdiffTime_last'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['bdiffTime_last'] = data['bdiffTime_last'] - data['context_timestamp']
    del temp

    data.loc[~data.duplicated(subset=subset, keep=False), ['bdiffTime_first', 'bdiffTime_last']] = -1


    return data

def do_category_trick(data):


    subset = ['user_id','last_category_id']
    data['cmaybe'] = 0
    pos = data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'cmaybe'] = 1
    pos = (~data.duplicated(subset=subset, keep='first')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'cmaybe'] = 2
    pos = (~data.duplicated(subset=subset, keep='last')) & data.duplicated(subset=subset, keep=False)
    data.loc[pos, 'cmaybe'] = 3
    del pos

    # features_trans = ['cmaybe']
    # data = pd.get_dummies(data, columns=features_trans)
    # data['cmaybe_0'] = data['cmaybe_0'].astype(np.int8)
    # data['cmaybe_1'] = data['cmaybe_1'].astype(np.int8)
    # data['cmaybe_2'] = data['cmaybe_2'].astype(np.int8)
    # data['cmaybe_3'] = data['cmaybe_3'].astype(np.int8)

    temp = data[['context_timestamp','user_id','last_category_id']]
    temp = temp.drop_duplicates(subset=subset, keep='first')
    temp.rename(columns={'context_timestamp': 'cdiffTime_first'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['cdiffTime_first'] = data['context_timestamp'] - data['cdiffTime_first']
    del temp

    temp = data[['context_timestamp','user_id','last_category_id']]
    temp = temp.drop_duplicates(subset=subset, keep='last')
    temp.rename(columns={'context_timestamp': 'cdiffTime_last'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['cdiffTime_last'] = data['cdiffTime_last'] - data['context_timestamp']
    del temp

    data.loc[~data.duplicated(subset=subset, keep=False), ['cdiffTime_first', 'cdiffTime_last']] = -1

    return data

# 处理item_property_list特征,根据分词预测概率当作新特征拼接
def property_feature(feature,dataset):

    feature_df = feature[['item_property_list','instance_id']]
    dataset_df = dataset[['item_property_list','instance_id']]

    feature_length = feature_df.shape[0]

    all_df=pd.concat([feature_df,dataset_df],axis=0)
    count_vec = TfidfVectorizer()
    all_ip = count_vec.fit_transform(all_df['item_property_list'])


    feature_ip = all_ip[0:feature_length,:]
    dataset_ip = all_ip[feature_length:,:]

    lgb_train = lgb.Dataset(feature_ip, feature['is_trade'])

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 24,
        'max_depth': 4,
        'learning_rate': 0.2,
        'seed':0,
        'colsample_bytree':1.0,
        'subsample':0.8
    }

    lgbm = lgb.train(params,
                     lgb_train,
                     num_boost_round=200)

    dataset_df['property_prob'] =lgbm.predict(dataset_ip,num_iteration=lgbm.best_iteration)
    # print(log_loss(dataset['is_trade'],dataset_df['property_prob']))
    dataset = pd.merge(dataset,dataset_df,on=['instance_id','item_property_list'],how='left')

    return dataset

# 类似处理predict_category_property特征
def predict_feature(feature,dataset):

    feature_df = feature[['predict_category_property', 'instance_id']]
    dataset_df = dataset[['predict_category_property', 'instance_id']]

    feature_length = feature_df.shape[0]

    all_df = pd.concat([feature_df, dataset_df], axis=0)
    count_vec = TfidfVectorizer()
    all_ip = count_vec.fit_transform(all_df['predict_category_property'])

    feature_ip = all_ip[0:feature_length, :]
    dataset_ip = all_ip[feature_length:, :]

    lgb_train = lgb.Dataset(feature_ip, feature['is_trade'])

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 63,
        'max_depth': 4,
        'learning_rate': 0.1,
        'seed': 2018,
        'colsample_bytree': 0.3,
        'subsample': 0.8
    }

    lgbm = lgb.train(params,
                     lgb_train,
                     num_boost_round=200)

    dataset_df['predict_prob'] = lgbm.predict(dataset_ip, num_iteration=lgbm.best_iteration)
    # print(log_loss(dataset['is_trade'], dataset_df['predict_prob']))
    dataset = pd.merge(dataset, dataset_df, on=['instance_id', 'predict_category_property'], how='left')

    return dataset

# 商铺相似度特征
def get_shop_share_features(feature,dataset):

    # Nmb: 每个商铺每种品牌商品的数量
    t=feature[['item_id','item_brand_id','shop_id']]
    t['shop_brand_item_num_Nmb']=t['item_id']
    t=t.groupby(['item_brand_id','shop_id'])['shop_brand_item_num_Nmb'].agg('count').reset_index()
    dataset=pd.merge(dataset,t,on=['item_brand_id','shop_id'],how='left')

    # Nm:每个商铺的商品总数量
    t1=feature[['shop_id','item_id']]
    t1['shop_item_num_Nm'] = t1['item_id']
    t1 = t1.groupby('shop_id')['shop_item_num_Nm'].agg('count').reset_index()
    dataset = pd.merge(dataset, t1, on='shop_id', how='left')


    # Nb:每种品牌商品数量
    t2 = feature[['item_brand_id', 'item_id']]
    t2['brand_item_num_Nb'] = t2['item_id']
    t2 = t2.groupby('item_brand_id')['brand_item_num_Nb'].agg('count').reset_index()
    dataset = pd.merge(dataset, t2, on='item_brand_id', how='left')


    # Umb:每个商铺每种品牌购买量
    t3=feature[['item_brand_id','shop_id','is_trade']]
    t3['shop_brand_trade_times_Umb']=t3['is_trade']
    t3 = t3.groupby(['item_brand_id','shop_id'])['shop_brand_trade_times_Umb'].agg(lambda x:sum(x==1)).reset_index()
    dataset = pd.merge(dataset,t3,on=['item_brand_id','shop_id'],how='left')


    # Um:每个商铺购买量
    t4=feature[['shop_id','is_trade']]
    t4['shop_trade_times_Um']=t4['is_trade']
    t4=t4.groupby('shop_id')['shop_trade_times_Um'].agg(lambda x:sum(x==1)).reset_index()
    dataset=pd.merge(dataset,t4,on='shop_id',how='left')


    # Ub:每种品牌购买量
    t5=feature[['item_brand_id','is_trade']]
    t5['brand_trade_times_Ub']=t5['is_trade']
    t5 = t5.groupby('item_brand_id')['brand_trade_times_Ub'].agg(lambda x: sum(x == 1)).reset_index()
    dataset = pd.merge(dataset, t5, on='item_brand_id', how='left')


    dataset['shop_market_share_on_brand']=dataset['shop_brand_item_num_Nmb']/dataset['brand_item_num_Nb']
    dataset['shop_user_share_on_brand']=dataset['shop_brand_trade_times_Umb']/dataset['brand_trade_times_Ub']
    dataset['brand_market_share_with_shop']=dataset['shop_brand_item_num_Nmb']/dataset['shop_item_num_Nm']
    dataset['brand_user_share_with_shop']=dataset['shop_brand_trade_times_Umb']/dataset['shop_trade_times_Um']

    dataset = dataset.drop(['shop_trade_times_Um','brand_trade_times_Ub'],axis=1)
    return dataset

# 穿越特征:用户在此次点击后还会点击多少次
def user_reclick_times(df):
    dic = {}
    def _a(group):
        # 遍历每条数据之后的数据
        for idx in group.index:
            dic[group.loc[idx,'instance_id']] = []
            curr_time_stamp = group.loc[idx,'context_timestamp']
            dic[group.loc[idx,'instance_id']].append(len(group[group.context_timestamp<=curr_time_stamp])-1)# 因为包括自己，所以减1
    df.groupby(['user_id','item_id']).apply(_a)
    df_temp = pd.DataFrame(dic,index=['leak_user_final_cat_reclick']).T
    df_temp['instance_id'] = df_temp.index

    df_temp.reset_index(inplace=True,drop=True)

    df=pd.merge(df,df_temp,on='instance_id',how='left')
    return df



if __name__=="__main__":

    # dataset3,2,1: (61259, 32) (57418, 33) (63611, 33)
    # feature3,2,1:(328923, 33) (342432, 33) (357082, 33)

     split_data()
    print(dataset3.shape,  dataset2.shape, dataset1.shape, feature3.shape,feature2.shape, feature1.shape)
    
    dataset3.to_csv('data_csv/data3.csv', index=False)
    dataset2.to_csv('data_csv/data2.csv', index=False)
    dataset1.to_csv('data_csv/data1.csv', index=False)
    
    feature1.to_csv('data_csv/feature1.csv',index=False)
    feature2.to_csv('data_csv/feature2.csv', index=False)
    feature3.to_csv('data_csv/feature3.csv', index=False)


    data3 = pd.read_csv('data_csv/data3.csv')
    data2 = pd.read_csv('data_csv/data2.csv')
    data1 = pd.read_csv('data_csv/data1.csv')
    
    label_feature3 = get_label_feature(data3)
    label_feature3.to_csv('data_csv/label_feature3.csv',index=False)
    label_feature2=get_label_feature(data2)
    label_feature2.to_csv('data_csv/label_feature2.csv', index=False)
    label_feature1 = get_label_feature(data1)
    label_feature1.to_csv('data_csv/label_feature1.csv', index=False)

    feature3 = pd.read_csv('data_csv/feature3.csv')
    label_feature3=pd.read_csv('data_csv/label_feature3.csv')
    dataset3 = get_one_feature(feature3,label_feature3)
    dataset3 = get_two_feature(feature3,dataset3)
    dataset3 = get_three_feature(feature3,dataset3)
    # dataset3 = get_shop_share_features(feature3,dataset3)
    dataset3.drop_duplicates(inplace=True)
    dataset3.to_csv('data_csv/dataset3.csv',index=False)

    feature2 = pd.read_csv('data_csv/feature2.csv')
    label_feature2 = pd.read_csv('data_csv/label_feature2.csv')
    dataset2 = get_one_feature(feature2, label_feature2)
    dataset2 = get_two_feature(feature2, dataset2)
    dataset2 = get_three_feature(feature2, dataset2)
    # dataset2 = get_shop_share_features(feature2,dataset2)
    dataset2.drop_duplicates(inplace=True)
    dataset2.to_csv('data_csv/dataset2.csv', index=False)

    feature1 = pd.read_csv('data_csv/feature1.csv')
    label_feature1 = pd.read_csv('data_csv/label_feature1.csv')
    dataset1 = get_one_feature(feature1, label_feature1)
    dataset1 = get_two_feature(feature1, dataset1)
    dataset1 = get_three_feature(feature1, dataset1)
    # dataset1 = get_shop_share_features(feature1,dataset1)
    dataset1.drop_duplicates(inplace=True)
    dataset1.to_csv('data_csv/dataset1.csv', index=False)


    features=[ 'item_category_list', 'item_property_list', 'predict_category_property','day']
    
    feature3 = pd.read_csv('data_csv/feature3.csv')
    dataset3 = pd.read_csv('data_csv/dataset3.csv')
    dataset3 = property_feature(feature3,dataset3)
    dataset3 = predict_feature(feature3,dataset3)
    dataset3 = dataset3.drop(features,axis=1)
    final3 = doTrick(dataset3)
    final3 = doTrick2(final3)
    final3 = user_reclick_times(final3)
    final3 = do_brand_trick(final3)
    final3 = do_category_trick(final3)
    final3.to_csv('data_csv/final3_encode.csv',index=False)
    
    feature2 = pd.read_csv('data_csv/feature2.csv')
    dataset2 = pd.read_csv('data_csv/dataset2.csv')
    dataset2 = property_feature(feature2,dataset2)
    dataset2 = predict_feature(feature2,dataset2)
    dataset2 = dataset2.drop(features, axis=1)
    final2 = doTrick(dataset2)
    final2 = doTrick2(final2)
    final2 = user_reclick_times(final2)
    final2 = do_brand_trick(final2)
    final2 = do_category_trick(final2)
    final2.to_csv('data_csv/final2_encode.csv', index=False)
    
    feature1 = pd.read_csv('data_csv/feature1.csv')
    dataset1 = pd.read_csv('data_csv/dataset1.csv')
    dataset1 = property_feature(feature1, dataset1)
    dataset1 = predict_feature(feature1, dataset1)
    dataset1 = dataset1.drop(features, axis=1)
    final1 = doTrick(dataset1)
    final1 = doTrick2(final1)
    final1 = user_reclick_times(final1)
    final1 = do_brand_trick(final1)
    final1 = do_category_trick(final1)
    final1.to_csv('data_csv/final1_encode.csv', index=False)
    
    print(final1.shape,final2.shape,final3.shape)


    final3=pd.read_csv('data_csv/final3_encode.csv')
    final3.to_csv('data_csv/final3_encode.csv',index=False)

    final2 = pd.read_csv('data_csv/final2_encode.csv')
    print(final2)
    final2.to_csv('data_csv/final2_encode.csv', index=False)

    final1 = pd.read_csv('data_csv/final1_encode.csv')
    print(final1[final1['is_trade']==1])
    final1.to_csv('data_csv/final1_encode.csv', index=False)
