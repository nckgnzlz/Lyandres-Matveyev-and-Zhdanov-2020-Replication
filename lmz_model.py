"""
lmz_model.py
------------
Closed-form real-options valuation model from

    Lyandres, Matveyev & Zhdanov (2020), "Does the Market Correctly Value
    Investment Options?", Review of Finance 24(6), 1159-1201.

Notation (paper -> code)
    theta  : curvature of production function (returns to scale), 0 < theta < 1
    eta    : cost of buying/installing one unit of capital
    mu     : risk-neutral drift of demand shock x
    sigma  : volatility of demand shock x
    lam    : capital depreciation rate (lambda)
    r      : risk-free rate
    tau    : corporate tax rate (paper uses 35%)
    K      : capital stock (Compustat PPEGT)
    x      : demand shock, backed out as x = EBITDA / K**theta

All functions are vectorised with numpy broadcasting, so you can pass arrays
for firms and/or parameter grids.
"""
import numpy as np

TAU = 0.35


def beta1(theta, mu, sigma, lam, r):
    """Positive root of eq. (4):
       0.5*sigma^2*b*(b-1) + [mu - lam*(theta-1)]*b - (r+lam) = 0
    """
    a = 0.5 * sigma**2
    b = (mu - lam * (theta - 1.0)) - a
    c = -(r + lam)
    return (-b + np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)


def theta_upper_bound(mu, sigma, lam, r, tol=1e-10):
    """Largest theta satisfying theta < 1 - 1/beta1(theta) (GO value finite).
    beta1 depends on theta, so solve the fixed point by bisection."""
    lo = np.zeros(np.broadcast(mu, sigma, lam, r).shape) + 1e-6
    hi = np.ones_like(lo) * (1 - 1e-6)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        ok = mid < 1.0 - 1.0 / beta1(mid, mu, sigma, lam, r)
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    return lo


def investment_boundary(K, theta, eta, mu, sigma, lam, r, tau=TAU):
    """Eq. (3): firm invests when x reaches X(K)."""
    b1 = beta1(theta, mu, sigma, lam, r)
    return b1 / (b1 - 1.0) * (r - mu + lam * theta) * eta * K**(1.0 - theta) / ((1.0 - tau) * theta)


def target_capital(x, theta, eta, mu, sigma, lam, r, tau=TAU):
    """Inverse of eq. (3): capital K* at which x sits exactly on the boundary.
    NOTE: the paper's printed K* formula (Section 3.2.a) has (r - mu) in the
    denominator; inverting eq. (3) exactly gives (r - mu + lam*theta).  We use
    the exact inverse so the adjusted firm lands on the boundary."""
    b1 = beta1(theta, mu, sigma, lam, r)
    return ((b1 - 1.0) / b1 * (1.0 - tau) * theta * x / ((r - mu + lam * theta) * eta))**(1.0 / (1.0 - theta))


def value_AP(K, x, theta, mu, lam, r, tau=TAU):
    """Eq. (9): value of assets in place."""
    return (1.0 - tau) * x * K**theta / (r - mu + lam * theta)


def value_GO(K, x, theta, eta, mu, sigma, lam, r, tau=TAU):
    """Eq. (10): value of growth options.  Returns +inf/nan where the
    finiteness constraint theta < 1 - 1/beta1 is violated."""
    b1 = beta1(theta, mu, sigma, lam, r)
    d = r - mu + lam * theta
    expo = b1 * (1.0 - theta) - 1.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        # work in logs for numerical stability
        logC = (b1 * np.log((1.0 - tau) * theta / (b1 * d))
                + (b1 - 1.0) * np.log((b1 - 1.0) / eta))
        logGO = logC + b1 * np.log(x) - expo * np.log(K) - np.log(expo)
        go = np.exp(logGO)
    return np.where(expo > 0, go, np.inf)


def firm_value(K, x, theta, eta, mu, sigma, lam, r, tau=TAU, adjust=True,
               return_parts=False):
    """Theoretical firm value V(K, x) (eq. 5), with the paper's adjustment
    (eq. 8) for firms found above the investment boundary: they are assumed to
    invest immediately up to K*, and value is V(K*, x) - (K* - K)*eta.
    """
    K = np.asarray(K, float)
    x = np.asarray(x, float)
    if adjust:
        X = investment_boundary(K, theta, eta, mu, sigma, lam, r, tau)
        above = x > X
        Kstar = target_capital(x, theta, eta, mu, sigma, lam, r, tau)
        Keff = np.where(above, np.maximum(Kstar, K), K)
        cost = np.where(above, (Keff - K) * eta, 0.0)
    else:
        Keff, cost = K, 0.0
    ap = value_AP(Keff, x, theta, mu, lam, r, tau)
    go = value_GO(Keff, x, theta, eta, mu, sigma, lam, r, tau) - cost
    v = ap + go
    if return_parts:
        return v, ap, go
    return v


def firm_value_AP_only(K, x, theta, mu, lam, r, tau=TAU):
    """Eq. (14): counterfactual model with growth options shut down."""
    return value_AP(K, x, theta, mu, lam, r, tau)
