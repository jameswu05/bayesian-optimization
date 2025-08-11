import numpy as np
from scipy.stats import norm

def expectedImprovement(X, GPPosteriorMean, GPPosteriorSTD, f_best):
    mu = np.array(GPPosteriorMean)
    sigma = np.array(GPPosteriorSTD)
    sigma = np.maximum(sigma, 1e-10)
    delta = f_best - mu
    Z = delta / sigma
    e_i = delta * norm.cdf(Z) + sigma * norm.pdf(Z)
    e_i = np.maximum(e_i, 0.0)
    return e_i
