import numpy as np
import matplotlib.pyplot as plt

def RBFkernel(X1, X2, l=1.0, sigma_f=1.0):
    diff = np.subtract.outer(X1, X2)**2
    return (sigma_f**2) * np.exp(-diff / (2*l**2))

def prior_distribution(X, n_samples, l=1.0, sigma_f=1.0, mean_func=None):
    X = np.array(X)
    if mean_func is None:
        mean = np.zeroes(len(X))
    else:
        mean = mean_func(X)
    
    covMatrix = RBFkernel(X, X, l, sigma_f)
    dist = np.random.multivariate_normal(mean, covMatrix, n_samples)
    return dist

# Define input points
X = np.linspace(-5, 5, 5)

# Optional mean function, zero mean in this example
mean_func = lambda x: np.zeros_like(x)

# Sample 3 functions from prior
samples = prior_distribution(X, n_samples=5, mean_func=mean_func)
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
