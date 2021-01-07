import os 
import arviz as az 
import pymc3 as pm 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import warnings

# explore categorical distribution
def gen_model(y_target):

    # use pm module 
    t_model = pm.Model()

    # switch params 
    param1 = [ .3, .3, .4]
    param2 = [ .7, .2, .1]

    with t_model:

        p_male = pm.Beta('p_male', alpha=1, beta=1)

        # choose the correct params
        ismale = pm.Bernoulli( 'ismale', p=p_male)
        name_params = pm.math.switch( ismale >= 1, param1, param2)

        male = pm.Categorical( 'male', p=name_params, observed=y_target)
    
    return t_model 

if __name__ == "__main__":

    # synthesis data 
    size = 100 
    y_target = np.random.choice(3, size=size)
    t_model = gen_model( y_target)

    with t_model:
        trace = pm.sample( 1000, return_inferencedata=False)
        print( az.summary( trace, round_to=2))

