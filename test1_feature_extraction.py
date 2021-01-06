import numpy as np 
import pandas as pd

dir = ''

def data_summary( data):
    txt = '数据收集的被试数量: {} \n'.format( data.shape[0])
    txt += '数据收集的特征数量: {} \n'.format( data.shape[1])
    return txt 

def float_weights( data, filters):
    '''Normalize the weights
    '''
    for key in filters:
        cols = filters[key]
        for col in cols:
            to_float = lambda x: float(x[0]) / np.sum(range(1,8))
            weight_lst = [to_float(weight) for weight in data.iloc[ :, col+1]]
            data.iloc[ :, col+1] = weight_lst

def float_valence( data, filters):
    '''Normalize the valenece
    '''
    for key in filters:
        cols = filters[key]
        for col in cols:
            to_float = lambda x: 1 - (int(x[0]) <= 4)
            weight_lst = [to_float(weight) for weight in data.iloc[ :, col+2]]
            data.iloc[ :, col+2] = weight_lst

def topN_weighted( data, filters, top_n=10):
    '''Weighted top N features
    '''
    txt = '\n考虑权重的典型姓名，行为，特质的结果 （这个结果比较有参考价值）:\n'
    for key in filters:
        cols = filters[key]
        unique_vars_lst = []
        unique_weight_lst = []
        for col in cols:
            vars_lst = data.iloc[ :, col]
            weight_lst = data.iloc[ :, col+1]
            for var, weight in zip( vars_lst, weight_lst):
                if var in unique_vars_lst:
                    idx = unique_vars_lst.index(var)
                    unique_weight_lst[idx] += weight
                else:
                    unique_vars_lst.append(var)
                    unique_weight_lst.append(weight)
        # sort: 
        series = pd.Series( unique_weight_lst, index = unique_vars_lst).sort_values(ascending=False)
        txt += '{}: \n{} \n'.format(key, series[:top_n])
    return txt

def topN_weighed_valence( data, filters, top_n=10):
    txt = '\n考虑效价后的典型姓名，行为，特质的结果 （这个结果比较有参考价值）:\n'
    for key in filters:
        cols = filters[key]
        pos_vars_lst = []
        neg_vars_lst = []
        pos_weight_lst = []
        neg_weight_lst = []
        for col in cols:
            vars_lst = data.iloc[ :, col]
            weight_lst = data.iloc[ :, col+1]
            val_lst = data.iloc[ :, col+2]
            for var, weight, val in zip( vars_lst, weight_lst, val_lst ):
                if val: 
                    if var in pos_vars_lst:
                        idx = pos_vars_lst.index(var)
                        pos_weight_lst[idx] += weight
                    else:
                        pos_vars_lst.append(var)
                        pos_weight_lst.append(weight)
                else:
                    if var in neg_vars_lst:
                        idx = neg_vars_lst.index(var)
                        neg_weight_lst[idx] += weight
                    else:
                        neg_vars_lst.append(var)
                        neg_weight_lst.append(weight)
        
        # sort pos series
        pos_series = pd.Series( pos_weight_lst, index = pos_vars_lst).sort_values(ascending=False)
        txt += '对积极词语+{}： \n{} \n'.format(key, pos_series[:top_n])
        # sort neg series
        neg_series = pd.Series( neg_weight_lst, index = neg_vars_lst).sort_values(ascending=False)
        txt += '对消极词语+{}: \n{} \n'.format(key, neg_series[:top_n])
    return txt 

if __name__ == "__main__":

    ## Load data 
    raw_data = pd.read_csv( dir + 'data/data_1020.csv')
    summary = data_summary(raw_data)

    # normalize the weights (from 7 pts --> [0, 1])
    weight_filters = {
        '典型男性姓名': [ 6, 8, 10],
        '典型女性姓名': [ 12, 14, 16],
        '典型男性行为': [ 18, 21, 24],
        '典型女性行为': [ 27, 30, 33],
        '典型男性特质': [ 40, 43, 46],
        '典型女性特质': [ 49, 52, 55],
    }
    float_weights( raw_data, weight_filters)
    
    # normalize the weights (from 7 pts --> [0, 1])
    val_filters = {
        '典型男性行为': [ 18, 21, 24],
        '典型女性行为': [ 27, 30, 33],
        '典型男性特质': [ 40, 43, 46],
        '典型女性特质': [ 49, 52, 55],
    }
    float_valence( raw_data, val_filters)

    # extract weighed top N 
    results_top_n_weighted = topN_weighted( raw_data, weight_filters, top_n = 10) 

    # extract weighted top N for each valence
    results_top_n_weighted_val = topN_weighed_valence( raw_data, val_filters)

    to_txt = summary + results_top_n_weighted + results_top_n_weighted_val

    print(to_txt)