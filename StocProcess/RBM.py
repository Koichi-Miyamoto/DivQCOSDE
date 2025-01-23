import numpy as np
from scipy.stats import norm

def RBMTransProb(x, t, x0, t0, c, d, mu, sigma, n_terms):

    if t <= t0:
        raise ValueError("t must be greater than t0")

    density1 = 0.0
    dt = t - t0
    sigma2 = sigma ** 2

    for n in range(-n_terms, n_terms + 1):
        density1 += np.exp(2 * mu * n * (c - d) / sigma2 - (x + 2 * n * (d - c) - x0 - mu * dt) ** 2 / (2 * sigma2 * dt))

    for n in range(-n_terms, n_terms + 1):
        density1 += np.exp(-2 * mu * (n * d - (n + 1) * c + x0) / sigma2 - (2 * n * d - 2 * (n + 1) * c + x0 + x - mu * dt) ** 2 / (2 * sigma2 * dt))

    density1 /= (sigma * np.sqrt(2 * np.pi * dt))

    density2 = 0.0

    for n in range(n_terms):
        term1 = np.exp(2 * mu * (n * d - (n + 1) * c + x) / sigma2)
        term2 = 1 - norm.cdf((mu * dt + 2 * n * d - 2 * (n + 1) * c + x0 + x) / (sigma * np.sqrt(dt)))
        density2 -= term1 * term2

    for n in range(n_terms):
        term1 = np.exp(2 * mu * (n * c - (n + 1) * d + x) / sigma2)
        term2 = norm.cdf((mu * dt - 2 * (n + 1) * d + 2 * n * c + x0 + x) / (sigma * np.sqrt(dt)))
        density2 += term1 * term2

    density2 *= 2 * mu / sigma2

    return density1 + density2

def MakeRBMTransProbFunc(t, t0, c, d, mu, sigma, n_terms):

    return lambda x, x0: RBMTransProb(x, t, x0, t0, c, d, mu, sigma, n_terms)
