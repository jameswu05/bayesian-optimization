import numpy as np
import matplotlib.pyplot as plt
import sys
import os

GP_path = "/mnt/c/Users/James Wu/Documents/BO/bayes_opt/GaussianProcess"
sys.path.append(GP_path)

from GPPosterior import GPPosterior
import expectedImprovement

X_train = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
y_train = np.sin(X_train)
X_test = np.linspace(0, 5, 100)

mu, sigma = GPPosterior(X_train, y_train, X_test, kernel="RBF", 
                        kernel_params={'l': 1.0, 'sigma_f': 1.0},
                        sigma_y=1e-8, mean_func='linear', mu=0.0, 
                        beta=1.0, param_selection='mle')

f_best = np.max(y_train)
ei = expectedImprovement.expectedImprovement(X_test, mu, sigma, f_best)

plt.figure(figsize=(10, 6))

# Plot the GP mean
plt.subplot(2, 1, 1)
plt.title("GP Posterior and EI")
plt.plot(X_test, mu, 'b-', label="GP mean")
plt.fill_between(X_test.ravel(),
                 mu - 1.96 * sigma,
                 mu + 1.96 * sigma,
                 color='blue', alpha=0.2, label="95% CI")
plt.scatter(X_train, y_train, c='red', marker='x', label="Data")
plt.legend()

# Plot EI
plt.subplot(2, 1, 2)
plt.plot(X_test, ei, 'g-', label="Expected Improvement")
plt.xlabel("x")
plt.ylabel("EI")
plt.legend()

plt.tight_layout()
plt.show()