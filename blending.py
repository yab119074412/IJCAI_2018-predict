#!/usr/bin/python
#-*- coding:utf-8 -*-

import  pandas as pd
import numpy as np


def averageBlending(a, b):
    sub1 = pd.read_csv('C:/Users/user/Desktop/ijcai_result/result_xgb_test.txt',sep=' ')
    sub2 = pd.read_csv('C:/Users/user/Desktop/ijcai_result/result_lgbm_test.txt',sep=' ')

    p1 = sub1.predicted_score.values
    p2 = sub2.predicted_score.values

    p1 = p1 * a
    p2 = p2 * b

    p = p1 + p2

    instanceID = sub1.instance_id.values

    output = pd.DataFrame({'instance_id': instanceID, 'predicted_score': p})
    print(output)
    output = output.iloc[18371:,:]
    output.to_csv('C:/Users/user/Desktop/ijcai_result/result_20180423.txt',sep=' ',index=False)


def toExpected():
    df_sub = pd.read_csv('C:/Users/user/Desktop/ijcai_result/submit.txt',sep=' ')
    mean = np.mean(df_sub.predicted_score.values)
    p = 0.0273 / mean
    df_sub['predicted_score'] = df_sub.predicted_score.apply(lambda x: x * p)
    df_sub.to_csv('C:/Users/user/Desktop/ijcai_result/result_20180423.txt',sep=' ',index=False)


def blend1():
    df_lgb = pd.read_csv('result/submission_lgb.csv')
    df_x2l = pd.read_csv('result/submission_x2l.csv')
    df_l2x = pd.read_csv('result/submission_l2x.csv')

    p_lgb = df_lgb.prob.values
    p_x2l = df_x2l.prob.values
    p_l2x = df_l2x.prob.values

    p_lgb = p_lgb * 0.5
    p_x2l = p_x2l * 0.5
    p_lgb_x2l = p_lgb + p_x2l

    p_lgb_x2l = p_lgb_x2l * 0.5
    p_l2x = p_l2x * 0.5

    p_fin = p_lgb_x2l + p_l2x

    instanceID = df_lgb.instanceID.values
    output = pd.DataFrame({'instanceID': instanceID, 'prob': p_fin})
    output.to_csv('result/submission3.csv', index=False)


def blend2():
    df_lgb = pd.read_csv('result/submission_lgb.csv')
    df_x2l = pd.read_csv('result/submission_x2l.csv')
    df_l2x = pd.read_csv('result/submission_l2x.csv')

    p_lgb = df_lgb.prob.values
    p_x2l = df_x2l.prob.values
    p_l2x = df_l2x.prob.values

    p_fin = p_lgb + p_x2l + p_l2x
    p_fin = p_fin / 3

    instanceID = df_lgb.instanceID.values
    output = pd.DataFrame({'instanceID': instanceID, 'prob': p_fin})
    output.to_csv('result/submission3.csv', index=False)


if __name__ == '__main__':
    averageBlending(0.7, 0.3)
    # toExpected()

    # blend2()
    # toExpected()