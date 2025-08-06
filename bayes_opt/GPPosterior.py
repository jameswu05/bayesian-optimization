import numpy as np
import kernels

def GPPosterior(X_train, y_train, X_test, kernel="RBF", kernel_params=None, sigma_y=1e-8, mean_func=None):
    X_train = np.array(X_train) # training input points, e.g [x1, x2, ..., xn]
    y_train = np.array(y_train) # observed function values at those points, e.g [f(x1), f(x2), ..., f(xn)]
    X_test = np.array(X_test) # points where you want to predict the function

    if mean_func is None:
        mu_train = np.zeros_like(y_train)
        mu_test = np.zeros_like(X_test)
    else:
        mu_train = mean_func(X_train)
        mu_test = mean_func(X_test)

    if kernel_params is None:
        kernel_params = {}

    if kernel == 'Gaussian' or kernel == 'Matern':
        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)

    if kernel == 'RBF':
        l = kernel_params.get('l', 1.0)
        sigma_f = kernel_params.get('sigma_f', 1.0)
        K = kernels.RBFkernel(X_train, X_train, l, sigma_f) + sigma_y**2 * np.eye(len(X_train))
        K_s = kernels.RBFkernel(X_test, X_train, l, sigma_f)
        K_ss = kernels.RBFkernel(X_test, X_test, l, sigma_f)
    elif kernel == 'Gaussian':
        alpha0 = kernel_params.get('alpha0', 1.0)
        alpha = kernel_params.get('alpha', None)
        K = kernels.GaussianKernel(X_train, X_train, alpha0, alpha) + sigma_y**2 * np.eye(len(X_train))
        K_s = kernels.GaussianKernel(X_test, X_train, alpha0, alpha)
        K_ss = kernels.GaussianKernel(X_test, X_test, alpha0, alpha)
    elif kernel == 'Matern':
        alpha0 = kernel_params.get('alpha0', 1.0)
        nu = kernel_params.get('nu', 1.5)
        length_scale = kernel_params.get('length_scale', 1.0)
        K = kernels.MaternKernel(X_train, X_train, alpha0, nu, length_scale) + sigma_y**2 * np.eye(len(X_train))
        K_s = kernels.MaternKernel(X_test, X_train, alpha0, nu, length_scale)
        K_ss = kernels.MaternKernel(X_test, X_test, alpha0, nu, length_scale)
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    L = np.linalg.cholesky(K)
    v = np.linalg.solve(L, y_train - mu_train)
    v_1 = np.linalg.solve(L, K_s.T)
    alpha = np.linalg.solve(L.T, v)

    mu_post = K_s @ alpha + mu_test
    cov_post = K_ss - v_1.T @ v

    return mu_post, cov_post