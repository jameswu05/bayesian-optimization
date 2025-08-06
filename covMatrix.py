import numpy as np
import matplotlib.pyplot as plt

# Define RBF kernel
def rbf_kernel(x, l=1.0):
    x = np.atleast_2d(x).T
    sqdist = (x - x.T)**2
    return np.exp(-0.5 * sqdist / l**2)

# Input points
x = np.linspace(-2, 2, 5)
K = rbf_kernel(x, l=1.0)

# Plot the covariance matrix
plt.figure(figsize=(6, 5))
plt.imshow(K, cmap='viridis', interpolation='none')
plt.colorbar(label='Covariance')
plt.title('Covariance Matrix (RBF Kernel)')
plt.xlabel('Index j')
plt.ylabel('Index i')
plt.xticks(range(len(x)), [f"x{j+1}" for j in range(len(x))])
plt.yticks(range(len(x)), [f"x{i+1}" for i in range(len(x))])
plt.grid(False)
plt.show()
