import numpy as np


def logit_margin(z, t):
    """Margin m = z_t - log sum_{j != t} exp(z_j), computed stably; returns a Python float."""
    z_t = z[t]
    z_new = np.delete(z,t)
    max_z = np.max(z_new)

    return float(z_t - max_z - np.log(np.sum(np.exp(z_new-max_z))))


def precision_cliff(dtype):
    """Smallest m >= 0 at which 1/(1 + exp(-m)) == 1 when evaluated in dtype; bisection on a float64 m.

    Returns the margin rounded to 4 decimals.
    """
    l = 0
    r = 100
    while (r-l) > 1e-9:
        median = np.median([l,r])
        denom = dtype(1) + dtype(np.exp(-median))
        div = np.divide(dtype(1.0),denom)
        if div == dtype(1.0):
            r = median
        else:
            l = median
    return float(np.round(median, decimals = 4))

