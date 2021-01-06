'''
Self-teach pymc3 
@author Zeming Fang
'''
import os 
import arviz as az 
import pymc3 as pm 
import numpy as np 
import matplotlib.pyplot as plt 

# current dir
dir = os.path.dirname(os.path.abspath(__file__))

# initialize random number generator
random_seed = 8927
np.random.seed( random_seed)
az.style.use( 'arviz-darkgrid')

def get_train_data( ):
    # True parameter values
    alpha, sigma = 1, 1
    beta = [ 1, 2.5]

    # size of dataset 
    size = 100 

    # predictor variable
    x1 = np.random.randn( size)
    x2 = np.random.randn( size) * .2 

    # simulate the evidence
    y = alpha + beta[0] * x1 + beta[1] * x2 + np.random.randn( size) * sigma

    return (x1, x2), y

def build_gen_model( x_input, y_target):
    
    # use pm module 
    gen_model = pm.Model()

    with gen_model: 

        # priors for unknown model parameters
        alpha = pm.Normal( 'alpha', mu=0, sigma=10)
        beta  = pm.Normal( 'beta', mu=0, sigma=10, shape=2)
        sigma = pm.HalfNormal( 'sigma', sigma=1)

        # predictor variable
        x1, x2 = x_input 

        # expected value of outcome
        mu = alpha + beta[0] * x1 + beta[1] * x2

        # likelihood
        y_obs = pm.Normal( 'y_obs', mu=mu, sigma=sigma, observed=y_target)

    return gen_model 


if __name__ == "__main__":

    # synthesize data
    x_input, y_target = get_train_data( )

    # build a generative model 
    gen_model = build_gen_model( x_input, y_target)

    # MAP inference
    # map_est = pm.find_MAP( model=gen_model)
    # print( map_est)

    # MCMC sampling inference
    # with gen_model:

    #     # draw 500 posterior samples 
    #     trace = pm.sample( 500, return_inferencedata=False)

    #     # posterior analysis
    #     fname = dir + '/results/tutorial_exp1.csv'
    #     az.summary( trace, round_to=2).to_csv( fname)
    


        


    
