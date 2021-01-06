'''
Code for gender counter-stereotype
@author Zeming Fang
'''

################################
###         Learning        ####
################################

import os 
import numpy as np 
import pandas as pd
import pickle 
from test1_feature_extraction import float_weights

dir = os.path.dirname(os.path.abspath(__file__))


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
        cat[mkey] = (mcats + prior) / np.sum( mcats + prior)
        cat[fkey] = (fcats + prior) / np.sum( fcats + prior)
    return cat

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
    
    # save the learned distribution 
    with open( dir + '/data/cat_dists.pkl', 'wb') as handle: 
        pickle.dump( cat, handle)


