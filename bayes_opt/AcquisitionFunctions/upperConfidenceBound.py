import numpy as np

def UCB(X, GPPosteriorMean, GPPosteriorSTD, l):
    mu = np.array(GPPosteriorMean)
    sigma = np.array(GPPosteriorSTD)
    return mu + l * sigma