"""
Generalized VAMP for the absolute value channel
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc, erfcx


def _log_erfc(x):
    """ Compute log(erfc(x)), using expansion for large x
    """
    if np.isscalar(x):
        x = np.array([x])
    res = np.empty(len(x))

    # For small x
    flag = (x < 20)
    res[flag] = np.log(erfc(x[flag]))

    # For large x
    _x = x[~flag]
    corr = -.5 / _x ** 2 + .75 / _x ** 4 - 1.875 / _x ** 6 + \
	6.5625 / _x ** 8 - 29.53125 / _x ** 10
    res[~flag] = -_x ** 2 - np.log(_x) - .5 * np.log(np.pi) + np.log1p(corr)

    return res


# Priors
def prior_gb(A, B, prmts):
    rho, mu, sig = prmts

    m = (B * sig + mu) / (1. + A * sig)
    v = sig / (1 + A * sig)
    p_s = rho / ( rho + (1 - rho) * np.sqrt(1. + A * sig) *
        np.exp(-.5 * m ** 2 / v + .5 * mu ** 2 / sig) )

    a = p_s * m
    c = p_s * v + p_s * (1. - p_s) * m ** 2
    return a, np.mean(c)


def prior_pm1(A, B, prmts):
    rho, = prmts

    a = np.tanh(.5 * np.log(rho / (1. - rho)) + B)
    c = 1. - a ** 2

    return a, np.mean(c)


# Channels
def channel_awgn(y, w, v, prmts):
    var_noise, = prmts

    g = (y - w) / (var_noise + v)
    dg = -1. / (var_noise + v)
    
    return g, dg


def channel_probit(y, w, v, prmts):
    var_noise, = prmts

    phi = -y * w / np.sqrt(2 * (var_noise + v))
    g = 2 * y / (np.sqrt(2 * np.pi * (var_noise + v)) * erfcx(phi))
    dg = -g * (w / (var_noise + v) + g)

    return g, dg


def channel_abs_zero(y, w, v, prmts):
    _, kappa = prmts
    if np.isscalar(y):
        y = np.array([y])

    t = np.tanh(y * w / v)
    g = (y * t - w) / v
    dg = (y / v) ** 2 * (1 - t ** 2) - 1 / v

    # Florent's trick no. 1
    flag = y < kappa
    g[flag] = (y - w)[flag] / v
    dg[flag] = -1 / v

    return g, dg


def channel_abs(y, w, v, prmts):
    var_noise, kappa = prmts
    v_eff = var_noise + v
    sig = np.sqrt(v_eff * (2 + var_noise / v) / (var_noise * v))

    w_sp = -((y + kappa) / var_noise + (w + kappa) / v) / (np.sqrt(2) * sig)
    w_sm = -((y - kappa) / var_noise - (w + kappa) / v) / (np.sqrt(2) * sig)

    # Compute p_+ = z_+ / (z_+ + z_-)
    logit = -.5 * ((y - w) ** 2 - (y + w) ** 2) / v_eff - \
            (_log_erfc(w_sm) - _log_erfc(w_sp))
    p = 1 / (1 + np.exp(-logit))

    # Compute g_+ and g_-
    g_p = (y - w) / v_eff - np.sqrt(2 / np.pi) / (v * sig * erfcx(w_sp))
    g_m = -(y + w) / v_eff + np.sqrt(2 / np.pi) / (v * sig * erfcx(w_sm))

    dg_p = -1 / v_eff - np.sqrt(2) * (2 / np.pi) / (v * sig) ** 2 * \
            (1 / erfcx(w_sp) - w_sp) / erfcx(w_sp)
    dg_m = 1 / v_eff + np.sqrt(2) * (2 / np.pi) / (v * sig) ** 2 * \
            (1 / erfcx(w_sm) - w_sm) / erfcx(w_sm)

    g = p * g_p + (1 - p) * g_m
    dg = p * (1 - p) * (g_p - g_m) ** 2 + p * dg_p - (1 - p) * dg_m

    return g, dg


# Functions to compute mean and variance of multivariate Gaussian
def lmmse(X, y, var, A, B):
    """Evaluate mean and variance of multivariate Gaussian given X"""

    # Some pre-computations
    A_r = X.T.dot(X) / var + A * np.eye(X.shape[1])
    B_r = X.T.dot(y) / var + B

    # Mean and variance
    a = np.linalg.solve(A_r, B_r)
    # c = np.trace(np.linalg.inv(A_r)) / X.shape[1]
    C = np.linalg.inv(A_r)

    return a, C


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
    c = np.sum(1. / ((S ** 2 / var)[:n] + A)) / n

    return a, c


# Solver
def gvamp(X, y,
          prior=prior_gb, prior_prmts=None,
          channel=channel_awgn, channel_prmts=None,
          true_coef=None,
          max_iter=250, tol=1e-7, verbose=1):
    """Iterate generalized VAMP equations"""
    n_samples, n_features = np.shape(X)

    # Initialize variables
    B1 = np.zeros(n_features)
    a1 = np.random.randn(n_features)
    A1 = 0.
    c1 = 1.

    B2 = np.zeros(n_features)
    a2 = np.zeros(n_features)
    A2 = 1.
    c2 = 1.

    Bz = np.random.randn(n_samples)
    y_eff = np.zeros(n_samples)
    Az = 1.
    var_eff = 1.

    # Compute SVD of X
    U, S0, V = np.linalg.svd(X, full_matrices=True)
    S = np.r_[S0, np.zeros(np.abs(V.shape[0] - U.shape[0]))]
    svd_X = (U, S, V)

    for t in range(max_iter):
        a1_old = np.copy(a1)

        # Estimates on z
        w, v = Bz / Az, 1. / Az
        g, dg = channel(y, w, v, channel_prmts)

        # dg = -np.mean(g ** 2) * np.ones(len(g))  # Florent's trick no. 2
        dg_mean = np.mean(dg)

        y_eff = w - g / dg_mean
        var_eff = -v - 1. / dg_mean

        # Estimates on x from likelihood
        a2, c2 = lmmse_svd(svd_X, y_eff, var_eff, A2, B2)
        # a2, C2 = lmmse(X, y_eff, var_eff, A2, B2)
        # c2 = np.trace(C2) / X.shape[1]

        # Messages/estimates on x from prior
        A1 = 1. / c2 - A2
        B1 = a2 / c2 - B2
        a1, c1 = prior(A1, B1, prior_prmts)

        # Messages on x from likelihood
        A2 = 1. / c1 - A1
        B2 = a1 / c1 - B1

        # Go from x to z: match 1st and 2nd moments
        az = X.dot(a2)
        cz = var_eff * np.sum(S0 ** 2 / (S0 ** 2 + A2 * var_eff)) / X.shape[0]
        # cz = np.trace(X.dot(C2).dot(X.T)) / X.shape[0]

        # Messages on z
        Az_ = 1. / cz - 1. / var_eff
        Bz_ = az / cz - y_eff / var_eff
        Az = .9 * Az + .1 * Az_  # NOTE: damping required
        Bz = .9 * Bz + .1 * Bz_

        if any(np.isnan(a1)):
            raise ValueError("NaNs, dg = %g" % (dg_mean))

        conv = np.mean((a1 - a1_old) ** 2)
        # mse = np.mean((a1 - true_coef) ** 2) if true_coef is not None else 0.
        if true_coef is not None:  # NOTE: sign symmetry
            mse = min(np.mean((a1 - true_coef) ** 2), np.mean((a1 + true_coef) ** 2))
        else:
            mse = 0.
        if verbose > 0:
            print("t = %d; conv = %g, mse = %g" % (t, conv, mse))

        if conv < tol:
            break

    return a1


def run_si(n_features, samples_to_features, frac_nonzeros, var_noise,
           kappa):
    n_samples = int(np.ceil(samples_to_features * n_features))
    n_nonzeros = int(np.ceil(frac_nonzeros * n_features))

    X = np.random.randn(n_samples, n_features) / np.sqrt(n_features)
    w = np.zeros(n_features)
    supp = np.random.choice(n_features, n_nonzeros, replace=False)
    w[supp] = np.random.randn(n_nonzeros)

    def f(z):
        res = np.copy(z)
        res[res < -kappa] *= -1
        return res
    y = f(X.dot(w) + np.sqrt(var_noise) * np.random.randn(n_samples))

    w_hat = gvamp(X, y,
                  prior=prior_gb, prior_prmts=(frac_nonzeros, 0., 1.),
                  channel=channel_abs, channel_prmts=(var_noise, kappa),
                  true_coef=w)


run_si(n_features=2000,
       samples_to_features=1.0,
       frac_nonzeros=0.3,
       var_noise=1e-8,
       kappa=0.)
