import numpy as np
import matplotlib.pyplot as plt
import argparse
import kernels

def prior_distribution(X, n_samples, mean_func=None, kernel='RBF', kernel_params=None):
    X = np.array(X)
    if mean_func is None:
        mean = np.zeros(len(X))
    else:
        mean = mean_func(X)

    if kernel_params is None:
        kernel_params = {}
    
    if kernel == 'RBF':
        # 1D only
        covMatrix = kernels.RBFkernel(X, X,
                                      l=kernel_params.get('l', 1.0),
                                      sigma_f=kernel_params.get('sigma_f', 1.0))
    elif kernel == 'Gaussian':
        # Must reshape for GaussianKernel
        X = X.reshape(-1, 1)
        covMatrix = kernels.GaussianKernel(X, X,
                                           alpha0=kernel_params.get('alpha0', 1.0),
                                           alpha=kernel_params.get('alpha', None))
    elif kernel == 'Matern':
        # Must reshape for MaternKernel
        X = X.reshape(-1, 1)
        covMatrix = kernels.MaternKernel(X, X,
                                         alpha0=kernel_params.get('alpha0', 1.0),
                                         nu=kernel_params.get('nu', 1.5),
                                         length_scale=kernel_params.get('length_scale', 1.0))
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    dist = np.random.multivariate_normal(mean, covMatrix, n_samples)
    return dist, X.flatten()

def main():
    parser = argparse.ArgumentParser(description="Sample from GP prior with different kernels.")
    parser.add_argument("--kernel", type=str, choices=["RBF", "Matern", "Gaussian"], default="RBF", help="Kernel to use.")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples to draw.")
    parser.add_argument("--n_points", type=int, default=50, help="Number of input points.")

    # For RBF kernel
    parser.add_argument("--l", type=float, default=1.0, help="Length scale for RBF kernel.")
    parser.add_argument("--sigma_f", type=float, default=1.0, help="Signal std dev for RBF kernel.")

    # For Matern kernel
    parser.add_argument("--length_scale", type=float, default=1.0, help="Length scale for Matern kernel.")
    parser.add_argument("--alpha0", type=float, default=1.0, help="Scale parameter alpha0 for Matern and Gaussian kernels.")
    parser.add_argument("--nu", type=float, default=1.5, help="Smoothness parameter for Matern kernel.")

    # For Gaussian kernel
    parser.add_argument("--alpha", nargs='+', type=float, help="List of alpha values for anisotropic Gaussian kernel.")

    args = parser.parse_args()

    # Define input points
    X = np.linspace(-5, 5, 5)

    # Optional mean function, zero mean in this example
    mean_func = lambda x: np.zeros_like(x)

    kernel_params = {}
    if args.kernel == "RBF":
        kernel_params = {"l": args.l, "sigma_f": args.sigma_f}
    elif args.kernel == "Gaussian":
        kernel_params = {"alpha0": args.alpha0, "alpha": np.array(args.alpha) if args.alpha else None}
    elif args.kernel == "Matern":
        kernel_params = {"alpha0": args.alpha0, "nu": args.nu, "length_scale": args.length_scale}

    # Sample 3 functions from prior
    samples, X_plot = prior_distribution(X, n_samples=5, mean_func=mean_func, kernel=args.kernel, kernel_params=kernel_params)
    print(samples)

    # Plotting
    plt.figure(figsize=(8,5))
    for i in range(samples.shape[0]):
        plt.plot(X, samples[i], lw=1.5, label=f'Sample {i+1}')
    plt.title("Samples from GP Prior")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()