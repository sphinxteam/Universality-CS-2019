import torch

# Priors
def prior_gb(A, B, prmts):
    """Compute f_a and f_c for Gauss-Bernoulli prior"""

    rho, mu, sig = prmts

    m = (B * sig + mu) / (1. + A * sig)
    v = sig / (1 + A * sig)
    p_s = rho / ( rho + (1 - rho) * torch.sqrt(1. + A * sig) *
        torch.exp(-.5 * m ** 2 / v + .5 * mu ** 2 / sig) )

    a = p_s * m
    c = p_s * v + p_s * (1. - p_s) * m ** 2
    return a, torch.mean(c)


# Function to compute mean and variance of multivariate Gaussian
def lmmse_svd(svd, y, var, A, B):
    """Evaluate mean and variance of multivariate Gaussian given SVD of X"""

    # Load SVD
    U, S, V = svd
    m, n = U.shape[0], V.shape[0]
  
    # Some pre-computations
    y_t = S[:m] * (U.t()@(y))
    B_t = V@B
    V_t = V.t() / (S[:n] ** 2 + A * var)

    # Mean and variance
    a = V_t[:, :m]@(y_t[:n]) + var * V_t@(B_t)
    c = torch.sum(var / (S ** 2 + A * var)) / V.shape[1]

    return a, c


# Solver
def vamp(X, y, var_noise,
         prior=prior_gb, prior_prmts=None, true_coef=None,
         max_iter=250, tol=1e-13, verbose=1):
    """Iterate VAMP equations"""

    n_samples, n_features = X.shape

    # Initialize variables
    B1 = torch.zeros(n_features)
    A1 = 0.
    a1 = torch.randn(n_features)
    c1 = 1.

    B2 = torch.zeros(n_features)
    A2 = 0.
    a2 = torch.zeros(n_features)
    c2 = 1.

    # Compute SVD of X
    U, S, V =torch.svd(X,some=False)

    V=V.t() #carreful, torch send back V not V.t() !
    S = torch.cat((S,torch.zeros(abs(V.shape[0] - U.shape[0]))))
    svd_X = (U, S, V)

    for t in range(max_iter):
        a1_old = a1.clone()
        
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
        conv = torch.mean((a1 - a1_old) ** 2)
        mse = torch.mean((a1 - true_coef) ** 2) if true_coef is not None else 0.
        if verbose > 0:
            print("t = %d; conv = %g, mse = %g" % (t, conv, mse))
            
        if conv < tol:
            break
        
    return a1
    
