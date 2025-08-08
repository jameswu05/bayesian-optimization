import numpy as np
import kernels
import argparse
import parameterSelection
import meanFunctions

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

    K = kernels.computeCovarianceMatrix(X_train, X_train, kernel, kernel_params) + sigma_y * np.eye(len(X_train))
    K_s = kernels.computeCovarianceMatrix(X_test, X_train, kernel, kernel_params)
    K_ss = kernels.computeCovarianceMatrix(X_test, X_test, kernel, kernel_params) + 1e-8 * np.eye(len(X_test))
    
    L = np.linalg.cholesky(K)
    v = np.linalg.solve(L, y_train - mu_train)
    v_1 = np.linalg.solve(L, K_s.T)
    alpha = np.linalg.solve(L.T, v)

    mu_post = K_s @ alpha + mu_test
    cov_post = K_ss - v_1.T @ v_1

    return mu_post, cov_post

def main():
    parser = argparse.ArgumentParser(description="Compute GP Posterior")

    parser.add_argument("--kernel", choices=["RBF", "Gaussian", "Matern"], default="RBF", help="Kernel type")
    parser.add_argument("--n_train", type=int, default=10, help="Number of training points")
    parser.add_argument("--n_test", type=int, default=100, help="Number of test points")
    parser.add_argument("--sigma_y", type=float, default=1e-8, help="Observation noise variance")
    parser.add_argument("--param_selection", choices=["none", "mle", "map", "bayes"], default="none",
                        help="Hyperparameter selection method")
    parser.add_argument("--mean_function", choices=["None", "constant", "linear"], default="None",
                        help="Mean function type")
    parser.add_argument("--mu", type=float, default=0.0, help="Constant mean offset")
    parser.add_argument("--beta", type=str, default=1.0, help="Slope for linear mean")

    args = parser.parse_args()

    if args.beta is not None:
        if isinstance(args.beta, str):
            args.beta = [float(b) for b in args.beta.split(',')]

    # Generate training and test data
    X_train = np.linspace(-5, 5, args.n_train)
    y_train = np.sin(X_train) + np.random.normal(0, np.sqrt(args.sigma_y), size=args.n_train)
    X_test = np.linspace(-5, 5, args.n_test)

    # Default kernel parameters
    kernel_params = {"l": 1.0, "sigma_f": 1.0}

    # Hyperparameter selection
    if args.param_selection != "none":
        kernel_params = parameterSelection.optimizeParameters(X_train, y_train, args.kernel, args.param_selection)

    mean_func = meanFunctions.selectMeanFunction(args.mean_function, mu=args.mu, beta=args.beta)

    # Posterior computation
    mu_post, cov_post = GPPosterior(X_train, y_train, X_test, kernel=args.kernel,
                                     kernel_params=kernel_params, sigma_y=args.sigma_y)

    # Optional: plot or print
    import matplotlib.pyplot as plt
    samples = np.random.multivariate_normal(mu_post, cov_post, 3)

    plt.figure()
    plt.plot(X_train, y_train, "ro", label="Training data")
    plt.plot(X_test, mu_post, "b-", label="Posterior mean")
    for i in range(samples.shape[0]):
        plt.plot(X_test, samples[i], "--", label=f"Sample {i + 1}")
    plt.legend()
    plt.title("GP Posterior Samples")
    plt.show()


if __name__ == "__main__":
    main()