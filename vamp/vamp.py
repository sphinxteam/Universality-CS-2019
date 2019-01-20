#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


# Priors
def prior_gb(A, B, prmts):
    """Compute f_a and f_c for Gauss-Bernoulli prior"""

    rho, mu, sig = prmts

    m = (B * sig + mu) / (1. + A * sig)
    v = sig / (1 + A * sig)
    p_s = rho / ( rho + (1 - rho) * np.sqrt(1. + A * sig) *
        np.exp(-.5 * m ** 2 / v + .5 * mu ** 2 / sig) )

    a = p_s * m
    c = p_s * v + p_s * (1. - p_s) * m ** 2
    return a, np.mean(c)


def prior_pm1(A, B, prmts):
    """Compute f_a and f_c for +/-1 prior"""

    rho, = prmts

    a = np.tanh(.5 * np.log(rho / (1. - rho)) + B)
    c = 1. - a ** 2

    return a, np.mean(c)


# Functions to compute mean and variance of multivariate Gaussian
def lmmse(X, y, var, A, B):
    """Evaluate mean and variance of multivariate Gaussian given X"""

    # Some pre-computations
    A_r = X.T.dot(X) / var + A * np.eye(X.shape[1])
    B_r = X.T.dot(y) / var + B

    # Mean and variance
    a = np.linalg.solve(A_r, B_r)
    c = np.trace(np.linalg.inv(A_r)) / X.shape[1]

    return a, c


def lmmse_svd(svd, y, var, A, B):
    """Evaluate mean and variance of multivariate Gaussian given SVD of X"""

    # Load SVD
    U, S, V = svd
    m, n = U.shape[0], V.shape[0]
  
    # Some pre-computations
    y_t = S[:m] * U.T.dot(y)
    B_t = V.dot(B)
    V_t = V.T / (S[:n] ** 2 + A * var)

    # Mean and variance
    a = V_t[:, :m].dot(y_t[:n]) + var * V_t.dot(B_t)
    c = np.sum(var / (S ** 2 + A * var)) / V.shape[1]

    return a, c


# Solver
def vamp(X, y, var_noise,
         prior=prior_gb, prior_prmts=None, true_coef=None,
         max_iter=250, tol=1e-13, verbose=1):
    """Iterate VAMP equations"""

    n_samples, n_features = np.shape(X)

    # Initialize variables
    B1 = np.zeros(n_features)
    A1 = 0.
    a1 = np.random.randn(n_features)
    c1 = 1.

    B2 = np.zeros(n_features)
    A2 = 0.
    a2 = np.zeros(n_features)
    c2 = 1.

    # Compute SVD of X
    U, S, V = np.linalg.svd(X, full_matrices=True)
    S = np.r_[S, np.zeros(np.abs(V.shape[0] - U.shape[0]))]
    svd_X = (U, S, V)

    mses = np.zeros(max_iter)
    for t in range(max_iter):
        a1_old = np.copy(a1)

        # Messages/estimates on x from likelihood
        A2 = 1. / c1 - A1
        B2 = a1 / c1 - B1
        # a2, c2 = lmmse(X, y, var_noise, A2, B2)
        a2, c2 = lmmse_svd(svd_X, y, var_noise, A2, B2)

        # Messages/estimates on x from prior
        A1 = 1. / c2 - A2
        B1 = a2 / c2 - B2
        a1, c1 = prior(A1, B1, prior_prmts)

        # Compute metrics
        conv = np.mean((a1 - a1_old) ** 2)
        mses[t] = np.mean((a1 - true_coef) ** 2) if true_coef is not None else 0.
        if verbose > 0:
            print("t = %d; conv = %g, mse = %g" % (t, conv, mses[t]))

        if conv < tol:
            break

    return a1, mses[:t]


def run_experiment(n_features, frac_nonzeros, samples_to_features, var_noise,
                   seed=42):
    """Run experiment with synthetic data"""
    n_samples = int(np.ceil(samples_to_features * n_features))
    n_nonzeros = int(np.ceil(frac_nonzeros * n_features))

    # Generate instance by sampling from generative model
    np.random.seed(seed)

    X = np.random.randn(n_samples, n_features) / np.sqrt(n_features)
    w = np.zeros(n_features)
    supp = np.random.choice(n_features, n_nonzeros, replace=False)
    w[supp] = np.random.randn(n_nonzeros)
    y = X.dot(w) + np.sqrt(var_noise) * np.random.randn(n_samples)

    # Compute estimate of w using VAMP
    w_hat, mses = vamp(X, y, var_noise,
                       prior=prior_gb, prior_prmts=(frac_nonzeros, 0, 1),
                       true_coef=w)

    print("Final MSE: %g" % (np.mean((w - w_hat) ** 2)))

    return mses


def main():
    # Parameters
    n_features = 4000
    frac_nonzeros = 0.3
    samples_to_features = 0.5
    var_noise = 1e-8

    mses = run_experiment(n_features, frac_nonzeros, samples_to_features, 
                          var_noise)
    plt.semilogy(mses, "ko", mfc="w")
    plt.show()

if __name__ == "__main__":
    main()
