'''
Code for gender counter-stereotype
@author Zeming Fang
'''

################################
###         Learning        ####
################################

import os 
import numpy as np
from numpy.lib.function_base import _insert_dispatcher 
import pandas as pd
import pickle 
from collections import OrderedDict
from test1_feature_extraction import float_weights

dir = os.path.dirname(os.path.abspath(__file__))

class cat_map:

    def __init__( self, in_dict):
        self.dist = in_dict
        self.cat_keys = list(in_dict.keys())
    
    def str_to_idx( self, cat_str):
        return self.cat_keys.index( cat_str)

    def idx_to_str( self, cat_idx):
        return self.cat_keys[ cat_idx]

    def str_to_prob( self, cat_str):
        return self.dist[ cat_str]

    def idx_to_prob( self, cat_idx):
        return self.dist[ cat_idx]

def learn_dist( data, filters):
    '''Sorted all cat

    This function is created to return all possible
    categories for variables N and V. To facilliate 
    the future works, we return the sorted category
    list in terms of weights. 

    Inputs: 
        data: the data needed to be extracted

    Outputs:
        cat: which contains:
            'names'     - the name categories
            'behaviors' - the behavrior categories
            'traits'   - the trait categories
    '''

    # init storage to
    cat = dict() 

    for key in filters.keys():
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
        cat[key] = pd.Series( unique_weight_lst, index = unique_vars_lst).sort_values(ascending=False)
    
    # match the female and the male 
    key_lsts = [ ('male_names', 'female_names'), ('male_behaviors',
                 'female_behaviors'), ('male_traits', 'female_traits')]
    for key in key_lsts:
        # get male keys and female keys respectively
        mkey, fkey = key
        mcats = cat[mkey]
        fcats = cat[fkey]
        # fill the missing cat
        m_need_id = set(list(mcats.index) + list(fcats.index)) - set(mcats.index)
        f_need_id = set(list(mcats.index) + list(fcats.index)) - set(fcats.index)
        m_filled = pd.Series( np.zeros(len(m_need_id)), index=m_need_id)
        f_filled = pd.Series( np.zeros(len(f_need_id)), index=f_need_id)
        mcats = mcats.append(m_filled)
        fcats = fcats.append(f_filled)
        # learn the distribution using perks prior 1/(nV x nP)
        # where nv is the state size, and nP is the paranets
        # configuration. Learning using bayeisan method
        prior = 1 / (len(mcats) + len(fcats))
        norm_male = (mcats + prior) / np.sum( mcats + prior)
        norm_female = (fcats + prior) / np.sum( fcats + prior)
        sorted_ind = (norm_male + norm_female).sort_values( ascending=False).index
        cat[mkey] = cat_map(norm_male.reindex( sorted_ind))
        cat[fkey] = cat_map(norm_female.reindex( sorted_ind))

    return cat

def idx_cat( data, filters, cat):
    '''Assign idx to the str

    Inputs: 
        data: the data we need to make change on
        filters: a dict tells the column to index on
        cats: dict tells the str-idx map 

    Output:
        data: transformed data
    '''
    for key in filters:
        cols = filters[key]
        mapping = cat[key]
        for col in cols: 
            ind_lst = [ mapping.str_to_idx(in_str) 
                        for in_str in data.iloc[ :, col] ]
            data.iloc[ :, col] = ind_lst 
    return data 

if __name__ == "__main__":

    ## Load data 
    raw_data = pd.read_csv( dir + '/data/data_1020.csv')

    # find the lines and normalize the data 
    filters = {
        'male_names': [ 6, 8, 10],
        'female_names': [ 12, 14, 16],
        'male_behaviors': [ 18, 21, 24],
        'female_behaviors': [ 27, 30, 33],
        'male_traits': [ 40, 43, 46],
        'female_traits': [ 49, 52, 55],
    }
    float_weights( raw_data, filters)
    
    # extract state space for N and V variables
    cat = learn_dist( raw_data, filters)

    # translate cat str to cat idx 
    processed_data = idx_cat( raw_data, filters, cat)

    # save the learned distribution  
    with open( dir + '/data/cat_dists.pkl', 'wb') as handle: 
        pickle.dump( cat, handle)

    # save the floatize data 
    processed_data.to_csv( dir + '/data/processed_data.csv')


