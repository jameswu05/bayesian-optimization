import numpy as np

def RBFkernel(X1, X2, l=1.0, sigma_f=1.0):
    diff = np.subtract.outer(X1, X2)**2
    return (sigma_f**2) * np.exp(-diff / (2*l**2))