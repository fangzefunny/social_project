'''
Code for gender counter-stereotype
@author Zeming Fang
'''

import pymc3 as pm 
import numpy as np 
import matplotlib.pyplot as plt 

# gnerate the artificial data 
obs_y = 1#np.random.normal( .5, .35, 2000)

# model that data with a simple bayesian model 
with pm.Model() as exercise1:

    rho1 = pm.Beta( 'male-prob', alpha=1, beta=1)
    
    mal_data = pm.Categorical( 'N')
    female_data = 
    y = pm.Normal( 'y', mu=mu, sd=stdev, observed=obs_y)
    
if __name__ == "__main__":

    with exercise1:
        trace = pm.sample(1000, cores=2)

        pm.traceplot( trace, ['mu', 'stdev'])
        plt.show() 
    
