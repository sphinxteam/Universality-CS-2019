"""
Generalized VAMP
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from vamp import run_experiment


# <f_c> for Rademacher prior
def prior_binary(A):
    f = lambda z: np.exp(-.5 * z ** 2) / np.sqrt(2 * np.pi) * \
            (1 - np.tanh(A + np.sqrt(A) * z))
    return quad(f, -10, 10)[0]

# <f_c> for Gauss-Bernoulli prior
def c_gb(A, B, rho):
    m = B / (1 + A)
    v = 1 / (1 + A)
    p_s = rho / (rho + (1 - rho) * \
            np.exp(-.5 * B ** 2 / (1 + A)) * np.sqrt(1 + A))
    return p_s * v + p_s * (1 - p_s) * m ** 2

def prior_gb(rho):
    def _prior(A):
        f_z = lambda z: np.exp(-.5 * z ** 2) / np.sqrt(2 * np.pi) * \
                c_gb(A, np.sqrt(A) * z, rho)
        f_nz = lambda z: np.exp(-.5 * z ** 2) / np.sqrt(2 * np.pi) * \
                c_gb(A, np.sqrt(A) * np.sqrt(A + 1) * z, rho)
        return (1 - rho) * quad(f_z, -10, 10)[0] + rho * quad(f_nz, -10, 10)[0]
    return _prior

# <f_c> for Gaussian prior
def prior_gaussian(A):
    return 1 / (1 + A)


# Stieltjes transform for Marcenko-Pastur ensemble
def likelihood_mp(a2, var_noise, alpha):
    z = -a2 * var_noise
    t = 1 + z - alpha
    s = (-t - np.sqrt(t ** 2 - 4 * z)) / (2 * z)
    M
    return var_noise * s

# Stieltjes transform for empirical ensemble
def likelihood_emp(a2, var_noise, lamb):
    z = -a2 * var_noise
    s = np.mean(1 / (lamb - z))
    return var_noise * s


def iterate_se(alpha, var_noise, prior, max_iter=250, tol=1e-13, verbose=1):
    """Iterate VAMP state evolution for y = Fx, w/ F Gaussian iid"""

    # Initialize variables
    v1 = 1
    a1 = 1

    # Compute singular values of F^T F
    n_cols = 2000
    F = np.random.randn(int(np.ceil(alpha * n_cols)), n_cols) / np.sqrt(n_cols)
    lamb = np.linalg.svd(F.T.dot(F), compute_uv=False)
   
    mses = np.zeros(max_iter)
    for t in range(max_iter):
        v1_old = v1

        # Perform iteration
        v1 = prior(a1)
        if v1 > 0:
            a2 = 1 / v1 - a1
            v2 = likelihood_emp(a2, var_noise, lamb)
            a1 = 1 / v2 - a2
        else:
            a2 = a1 = np.inf
            v2 = 0

        # Compute metrics and print iteration status on screen
        mses[t] = v1
        diff = np.abs(v1 - v1_old)
        if verbose:
            print("t = %d, diff = %g, v1 = %g, v2 = %g" % (t, diff, v1, v2))
        if diff < tol or v1 == 0:
            break

    return mses[:t]


# Run state evolution and VAMP
print("Running state evolution...")
mses_se = iterate_se(alpha=0.5, var_noise=1e-8, prior=prior_gb(rho=0.3))

print("Running VAMP...")
mses_vamp = run_experiment(n_features=4000, frac_nonzeros=0.3,
                      samples_to_features=0.5, var_noise=1e-8)


# Plot results
plt.semilogy(mses_se, "k", lw=3, alpha=0.7, label="SE")
plt.semilogy(mses_vamp, "ko", mfc="w", label="VAMP")
plt.xlabel("iteration")
plt.ylabel("MSE")
plt.show()
