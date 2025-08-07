import numpy as np
from scipy.optimize import minimize
from kernels import RBFkernel, GaussianKernel, MaternKernel

def log_marginal_likelihood(theta, X, y, kernel_name, sigma_y=1e-8, prior_func=None):
    # Map theta vector to kernel params depending on kernel_name
    if kernel_name == 'RBF':
        l, sigma_f = theta
        if l <= 0 or sigma_f <= 0:
            return np.inf
        K = RBFkernel(X, X, l=l, sigma_f=sigma_f) + (sigma_y**2) * np.eye(len(X))

    elif kernel_name == 'Gaussian':
        # Example: assume theta = [alpha0, alpha1, alpha2, ...] where alpha1..d are anisotropic weights
        alpha0 = theta[0]
        alpha = theta[1:]
        if alpha0 <= 0 or np.any(alpha <= 0):
            return np.inf
        K = GaussianKernel(X, X, alpha0=alpha0, alpha=alpha) + (sigma_y**2) * np.eye(len(X))

    elif kernel_name == 'Matern':
        # Assume theta = [alpha0, nu, length_scale]
        alpha0, nu, length_scale = theta
        if alpha0 <= 0 or nu <= 0 or length_scale <= 0:
            return np.inf
        K = MaternKernel(X, X, alpha0=alpha0, nu=nu, length_scale=length_scale) + (sigma_y**2) * np.eye(len(X))

    else:
        raise ValueError(f"Unsupported kernel: {kernel_name}")

    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return np.inf

    alpha_vec = np.linalg.solve(L.T, np.linalg.solve(L, y))

    log_likelihood = -0.5 * y.T @ alpha_vec
    log_likelihood -= np.sum(np.log(np.diag(L)))
    log_likelihood -= (len(X) / 2) * np.log(2 * np.pi)

    if prior_func is not None:
        log_prior = prior_func(theta)
        return -(log_likelihood + log_prior)
    else:
        return -log_likelihood

def maximizeLikelihoodEstimation(X, y, kernel_name='RBF', initial_theta=None, sigma_y=1e-8):
    if kernel_name == 'RBF':
        if initial_theta is None:
            initial_theta = [1.0, 1.0]
        bounds = [(1e-5, 10), (1e-5, 10)]

    elif kernel_name == 'Gaussian':
        d = X.shape[1] if len(X.shape) > 1 else 1
        if initial_theta is None:
            initial_theta = [1.0] + [1.0] * d  # alpha0 + alpha vector
        bounds = [(1e-5, 10)] * (d + 1)

    elif kernel_name == 'Matern':
        if initial_theta is None:
            initial_theta = [1.0, 1.5, 1.0]  # alpha0, nu, length_scale
        bounds = [(1e-5, 10), (0.5, 5), (1e-5, 10)]

    else:
        raise NotImplementedError(f"MLE not implemented for kernel {kernel_name}")

    res = minimize(log_marginal_likelihood, initial_theta, args=(X, y, kernel_name, sigma_y, None),
                   bounds=bounds, method='L-BFGS-B')

    return res.x

def maximumAPosteriori(X, y, kernel_name='RBF', initial_theta=None, sigma_y=1e-8):
    def prior(theta):
        # Simple Gaussian prior on log params for positivity and regularization
        log_theta = np.log(theta)
        mu = 0.0
        sigma = 1.0
        return -0.5 * np.sum(((log_theta - mu) / sigma) ** 2)

    if initial_theta is None:
        if kernel_name == 'RBF':
            initial_theta = [1.0, 1.0]
        elif kernel_name == 'Gaussian':
            d = X.shape[1] if len(X.shape) > 1 else 1
            initial_theta = [1.0] + [1.0] * d
        elif kernel_name == 'Matern':
            initial_theta = [1.0, 1.5, 1.0]
        else:
            raise NotImplementedError(f"MAP not implemented for kernel {kernel_name}")

    bounds = [(1e-5, 10)] * len(initial_theta)
    if kernel_name == 'Matern':
        bounds[1] = (0.5, 5)  # nu range

    res = minimize(log_marginal_likelihood, initial_theta, args=(X, y, kernel_name, sigma_y, prior),
                   bounds=bounds, method='L-BFGS-B')

    return res.x

def optimizeParameters(X, y, kernel='RBF', param_selection='None', cli_params=None, sigma_y=1e-8):
    if param_selection == 'None':
        if cli_params is None:
            raise ValueError("cli_params dict must be provided when param_selection=None")
        
        if kernel == 'RBF':
            return {
                "l": cli_params.get("l", 1.0), 
                "sigma_f": cli_params.get("sigma_f", 1.0)
                }
        elif kernel == 'Gaussian':
            return {
                "alpha0": cli_params.get("alpha0", 1.0), 
                "alpha": cli_params.get("alpha", None)
                }
        elif kernel == 'Matern':
            return {
                "alpha0": cli_params.get("alpha0", 1.0),
                "nu": cli_params.get("nu", 1.5), 
                "length_scale": cli_params.get("length_scale", 1.0)
                }
        else:
            raise ValueError(f"Unsupported kernel: {kernel}")
        
    elif param_selection == 'mle':
        optim_params = maximizeLikelihoodEstimation(X, y, kernel_name=kernel, sigma_y=sigma_y)
    elif param_selection == 'map':
        optim_params = maximumAPosteriori(X, y, kernel_name=kernel, sigma_y=sigma_y)
    else:
        raise ValueError(f"Unsupported parameter selection method: {param_selection}")
    
    if kernel == "RBF":
        kernel_params = {"l": optim_params[0], "sigma_f": optim_params[1]}
    elif kernel == "Gaussian":
        kernel_params = {"alpha0": optim_params[0], "alpha": optim_params[1:]}
    elif kernel == "Matern":
        kernel_params = {"alpha0": optim_params[0], "nu": optim_params[1], "length_scale": optim_params[2]}
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")
    
    return kernel_params