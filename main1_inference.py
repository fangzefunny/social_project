'''
Code for gender counter-stereotype
@author Zeming Fang
'''

import os 
import arviz as az 
import pymc3 as pm 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pymc3.distributions.continuous import HalfNormal 

# build generative model
def get_model( y_target, params):
    '''Reason net for stereotype

    The probability model is: 
        p_male ~ Beta( b1, b2)
        ismale ~ Ber( p_male) 
        name ~ Cat( name_params[ismale])
        verb ~ Cat( verb_params[ismale])
    '''
    # load the evidence
    name_evid = y_target['name']
    verb_evid = y_target['verb']

    # load distribution
    name_dist = params['name']
    verb_dist = params['verb']

    # use the pm.module
    reason_net = pm.Model()

    # handcrafted the relationship
    with reason_net:

        # estimate the prior for the 
        alpha = HalfNormal( 'alpha', sigma=20)
        beta  = HalfNormal( 'beta', sigma=20)

        # sample male or female 
        p_male = pm.Beta( 'prior', alpha=alpha, beta=beta)

        # choose the correct params
        ismale = pm.Bernoulli( 'ismale', p_male)
        name_params = pm.math.switch( ismale == 1, name_evid['female'])
        verb_params = pm.math.switch( ismale == 1, verb_evid['female'])
        y_obs1 = pm.Categorical( 'name', p=name_params, observed=name_evid)
        y_obs2 = pm.Categorical( 'verb', p=verb_params, observed=verb_evid)



    
