import numpy as np

def parametric_mean(x, mu=0.0, beta=None, basis_funcs=None):
    x = np.asarray(x)
    result = np.full_like(x, fill_value=mu, dtype=np.float64)

    if beta is None or len(beta) == 0:
        return result
    
    beta = np.asarray(beta)
    if basis_funcs is None:
        basis_funcs = [lambda x, deg=i+1: x**deg for i in range(len(beta))]
    
    for b, psi in zip(beta, basis_funcs):
        result += b * psi(x)

    return result

def selectMeanFunction(mean_func=None, mu=0.0, beta=None):
    if beta is None:
        beta = []

    if mean_func == "None":
        mean_func = lambda x: parametric_mean(x, mu=0.0, beta=[])
        return mean_func
    elif mean_func == "constant":
        mean_func = lambda x: parametric_mean(x, mu=mu, beta=[])
        return mean_func
    elif mean_func == "linear":
        if isinstance(beta, float):
            beta_list = [beta]
        else:
            beta_list = beta
        mean_func = lambda x: parametric_mean(x, mu=mu, beta=beta_list)
        return mean_func
    else:
        mean_func = lambda x: parametric_mean(x, mu=0.0, beta=[])
        return mean_func