"""
Generalized VAMP state evolution
"""
from time import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import erfcx

import dnner
from dnner.priors import Bimodal, Normal
from dnner.activations import Probit, Linear
from dnner.ensembles import Gaussian
from dnner.utils.integration import H


# <f_c> for Rademacher prior
def prior_binary(A):
    f = lambda z: np.exp(-.5 * z ** 2) / np.sqrt(2 * np.pi) * \
            (1 - np.tanh(A + np.sqrt(A) * z))
    return quad(f, -10, 10)[0]

# <f_c> for Gaussian prior
def prior_gaussian(A):
    return 1 / (1 + A)

# <g^2> for probit likelihood
def channel_probit(v, rho=1, var_noise=1e-10):
    v_eff = var_noise + v
    v_int = .5 * (rho - v) / v_eff
    def f(z):
        return np.exp(-z ** 2) / erfcx(z)
    H_f = H(f, var=v_int)
    return (2 / np.pi) * H_f / v_eff

# <g^2> for Gaussian likelihood
def channel_awgn(v, rho=1, var_noise=1e-10):
    return 1 / (v + var_noise)

# Stieltjes transform for Marcenko-Pastur ensemble
def llmse_mp(a, var_noise, alpha):
    z = -a * var_noise
    t = 1 + z - alpha
    s = (-t - np.sqrt(t ** 2 - 4 * z)) / (2 * z)
    return var_noise * s

# Stieltjes transform for empirical ensemble
def llmse_emp(a, var_noise, lamb):
    z = -a * var_noise
    s = np.mean(1 / (lamb - z))
    return var_noise * s

def iterate_se(alpha, var_noise, prior, channel, spectrum=None,
               max_iter=250, tol=1e-7, verbose=1):
    """Iterate VAMP state evolution for y = sign(Fx), w/ F Gaussian iid"""

    # Initialize variables
    # NOTE: how to find good solution?
    a1x = 1
    a1z_i = 1

    v1x = 1

    if spectrum is None:
        llmse = lambda x, y: llmse_mp(x, y, alpha)
        g1z = lambda a: channel(1 / a, 
                rho=1, var_noise=var_noise)
    else:
        llmse = lambda x, y: llmse_emp(x, y, spectrum)
        g1z = lambda a: channel(1 / a, 
                rho=spectrum.mean() / alpha, var_noise=var_noise)
  
    # Iterate equations
    # NOTE: how to translate vars to free energy computation?
    mses = np.zeros(max_iter)
    for t in range(max_iter):
        v1x_old = v1x

        v1x = prior(a1x)  # compute variance on x_1
        if v1x > np.spacing(1):
            a2x = 1 / v1x - a1x  # message from x_1 to x_2
            v2x = llmse(a2x, a1z_i)
            v1z = a1z_i * (1 - a2x * v2x) / alpha  # compute variance on z_2
            a2z = 1 / v1z - 1 / a1z_i  # message from z_1 to z_2

            a1z_i = 1 / g1z(a2z) - 1 / a2z  # message from z_2 to z_1
            v2x = llmse(a2x, a1z_i)  # compute variance on x_2
            a1x = 1 / v2x - a2x  # message from x_2 to x_1
        else:
            a2x = a1x = np.inf
            v2x = v1x = 0

            a2z = np.inf
            a1z_i = v1z = 0

        # Compute metrics and print iteration status on screen
        v2z = 1 / a2z - g1z(a2z) / a2z ** 2

        mses[t] = v1x
        diff = np.abs(v1x - v1x_old)
        if verbose:
            print("t = %d, diff = %g, v1z = %g, v2z = %g, v1x = %g, v2x = %g" % \
                    (t, diff, v1z, v2z, v1x, v2x))
        if diff < tol or v1x == 0:
            break

    # Comput fixed points in our notation
    v = v1x
    a_in = a1x
    v_in = 1 / a2z
    a = g1z(a2z)
    theta = 1 / (1 + a2z * a1z_i)

    fps = [v, a, v_in, a_in, theta]

    return mses[:t + 1], fps


# Parameters
alpha = 0.4
var_noise = 1e-10

mses_vamp = []
mses_dnner = []

# Compute singular values of F^T F
n_cols = 2000
F = np.random.randn(int(np.ceil(alpha * n_cols)), n_cols) / np.sqrt(n_cols)
spectrum = np.linalg.svd(F.T.dot(F), compute_uv=False)

# Run G-VAMP state evolution
start_time = time()
mses, fps_vamp = iterate_se(alpha, var_noise, prior_binary, channel_probit,
        spectrum=spectrum, max_iter=250, tol=1e-7, verbose=1)
elapsed_vamp = time() - start_time

# Run dnner
layers = [Bimodal(0.5), Probit(var_noise)]
weights = [(alpha, spectrum)]
start_time = time()
e, x = dnner.compute_entropy(layers, weights, return_extra=True,
        v0=1, max_iter=250, tol=1e-7, verbose=1)
elapsed_dnner = time() - start_time

fps_dnner = [fp[0] for fp in x["fixed_points"]]

# Print results
print(" - final MSE from G-VAMP SE is %g (%gs)" % (mses[-1], elapsed_vamp))
print("   - fixed points are: %s" % (fps_vamp,))
print(" - final MSE from dnner is %g (%gs)" % (x["mmse"], elapsed_dnner))
print("   - fixed points are: %s" % (fps_dnner,))
