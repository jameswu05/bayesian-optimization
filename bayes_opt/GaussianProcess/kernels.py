import numpy as np
from scipy.special import kv, gamma

def RBFkernel(X1, X2, l=1.0, sigma_f=1.0):
    diff = np.subtract.outer(X1, X2)**2
    return (sigma_f**2) * np.exp(-diff / (2*l**2))

def GaussianKernel(X1, X2, alpha0=1.0, alpha=None):
    """
    Computes the Gaussian (RBF) kernel matrix between X1 and X2.
    
    Parameters:
    - X1: shape (n1, d)
    - X2: shape (n2, d)
    - alpha0: scaling factor (alpha_0)
    - alpha: list or array of length d for anisotropic scaling (alpha_1 to alpha_d). If None, isotropic with 1.0
    
    Returns:
    - K: kernel matrix of shape (n1, n2)
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    
    if alpha is None:
        alpha = np.ones(X1.shape[1])
    alpha = np.array(alpha)

    # Weighted squared distance
    dists = np.sum(alpha * (X1[:, np.newaxis, :] - X2[np.newaxis, :, :])**2, axis=2)
    return alpha0 * np.exp(-dists)

def MaternKernel(X1, X2, alpha0=1.0, nu=1.5, length_scale=1.0):
    """
    Computes the Matern kernel between points in X1 and X2.
    
    Parameters:
    - X1: shape (n1, d)
    - X2: shape (n2, d)
    - alpha0: scaling constant (α0)
    - nu: smoothness parameter (commonly 0.5, 1.5, 2.5, ∞)
    - length_scale: scalar or array of length d
    
    Returns:
    - Kernel matrix of shape (n1, n2)
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)

    if np.isscalar(length_scale):
        length_scale = np.full(X1.shape[1], length_scale)
    length_scale = np.array(length_scale)

    # Compute scaled Euclidean distance
    diff = (X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) / length_scale
    r = np.sqrt(np.sum(diff**2, axis=2))

    if nu == 0.5:
        # Exponential kernel
        K = alpha0 * np.exp(-r)
    else:
        factor = np.sqrt(2 * nu) * r
        coeff = alpha0 * (2 ** (1 - nu)) / gamma(nu)
        # Avoid zero times inf at r=0
        factor[factor == 0.0] += 1e-10
        K = coeff * (factor ** nu) * kv(nu, factor)

    return K

def computeCovarianceMatrix(X1, X2, kernel='RBF', kernel_params=None):
    if kernel_params is None:
        kernel_params = {}

    if kernel == 'RBF':
        l = kernel_params.get("l", 1.0)
        sigma_f = kernel_params.get("sigma_f", 1.0)
        return RBFkernel(X1, X2, l=l, sigma_f=sigma_f)
    elif kernel == 'Gaussian':
        alpha0 = kernel_params.get("alpha0", 1.0)
        alpha = kernel_params.get("alpha", None)
        return GaussianKernel(X1, X2, alpha0=alpha0, alpha=alpha)
    elif kernel == 'Matern':
        alpha0 = kernel_params.get("alpha0", 1.0)
        nu = kernel_params.get("nu", 1.5)
        length_scale = kernel_params.get("length_scale", 1.0)
        return MaternKernel(X1, X2, alpha0=alpha0, nu=nu, length_scale=length_scale)
    else:
        return ValueError(f"Unsupported kernel name: {kernel}")