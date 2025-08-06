import numpy as np
import GPPrior

def GPPosterior(X_train, y_train, X_test, l=1.0, sigma_f=1.0, sigma_y=1e-8, mean_func=None):
    X_train = np.array(X_train) # training input points, e.g [x1, x2, ..., xn]
    y_train = np.array(y_train) # observed function values at those points, e.g [f(x1), f(x2), ..., f(xn)]
    X_test = np.array(X_test) # points where you want to predict the function

    if mean_func is None:
        mu_train = np.zeros_like(y_train)
        mu_test = np.zeros_like(X_test)
    else:
        mu_train = mean_func(X_train)
        mu_test = mean_func(X_test)

    K = GPPrior.RBFkernel(X_train, X_train, l, sigma_f)
    K_s = GPPrior.RBFkernel(X_test, X_train, l, sigma_f)
    K_ss = GPPrior.RBFkernel(X_test, X_test, l, sigma_f)

    L = np.linalg.cholesky(K)
    v = np.linalg.solve(L, y_train - mu_train)
    v_1 = np.linalg.solve(L, K_s.T)
    alpha = np.linalg.solve(L.T, v)

    mu_post = K_s @ alpha + mu_test
    cov_post = K_ss - v_1.T @ v

    return mu_post, cov_post