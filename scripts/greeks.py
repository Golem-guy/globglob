"""
greeks.py

Black-Scholes-Merton pricing and Greeks, generalized with a continuous
dividend yield q (fed in from the futures-implied cost-of-carry computed
in nse_fetch.py / run_fetch.py -- this file has no opinion on where q
comes from).

DESIGN CHOICE on the higher-order Greeks (vanna, charm, vomma, speed,
zomma, color, veta): rather than hand-transcribing their closed-form
formulas -- which vary in sign convention across textbooks and are easy
to get subtly wrong -- these are computed via central finite differences
of the already-verified first-order formulas (delta, gamma, vega). This
trades a small amount of numerical approximation error for a large
reduction in the risk of a silent formula/sign bug. That's the right
trade for a research/backtesting pipeline; tune H_SIGMA / H_S_PCT below
if you ever see noisy results.

Units:
- T (time to expiry) is in YEARS.
- sigma (volatility) is in decimal form, e.g. 0.15 for 15% annualized vol.
- "Per day" quantities (theta, charm, color, veta) are the per-YEAR
  analytical/numerical result divided by 365 (calendar days) -- matching
  how time-to-expiry itself is computed elsewhere in this pipeline.
- vega, vomma, veta are expressed "per 1 percentage point of IV"
  (i.e. per 0.01 change in sigma), matching common trading-desk convention.
"""

import math

# ---------------------------------------------------------------------------
# Finite-difference bump sizes
# ---------------------------------------------------------------------------
H_SIGMA = 0.001          # absolute vol bump (0.001 = 0.1 vol point)
H_S_PCT = 0.005          # spot bump as a fraction of spot (0.5%)
DAYS_PER_YEAR = 365.0


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1_d2(S, K, T, r, q, sigma):
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


# ---------------------------------------------------------------------------
# Price
# ---------------------------------------------------------------------------

def bs_price(S, K, T, r, q, sigma, option_type):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        if option_type == "CE":
            return max(S - K, 0.0)
        return max(K - S, 0.0)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    if option_type == "CE":
        return S * math.exp(-q * T) * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * math.exp(-q * T) * _norm_cdf(-d1)


# ---------------------------------------------------------------------------
# First-order analytical Greeks
# ---------------------------------------------------------------------------

def bs_delta(S, K, T, r, q, sigma, option_type):
    if T <= 0 or sigma <= 0:
        return None
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    if option_type == "CE":
        return math.exp(-q * T) * _norm_cdf(d1)
    return math.exp(-q * T) * (_norm_cdf(d1) - 1.0)


def bs_gamma(S, K, T, r, q, sigma):
    if T <= 0 or sigma <= 0:
        return None
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return math.exp(-q * T) * _norm_pdf(d1) / (S * sigma * math.sqrt(T))


def bs_vega_per_1pct(S, K, T, r, q, sigma):
    """Price change per 1 percentage point (0.01) change in IV."""
    if T <= 0 or sigma <= 0:
        return None
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return S * math.exp(-q * T) * _norm_pdf(d1) * math.sqrt(T) / 100.0


def bs_theta_per_day(S, K, T, r, q, sigma, option_type):
    if T <= 0 or sigma <= 0:
        return None
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    pdf_d1 = _norm_pdf(d1)
    if option_type == "CE":
        theta_annual = (
            -S * math.exp(-q * T) * pdf_d1 * sigma / (2 * math.sqrt(T))
            - r * K * math.exp(-r * T) * _norm_cdf(d2)
            + q * S * math.exp(-q * T) * _norm_cdf(d1)
        )
    else:
        theta_annual = (
            -S * math.exp(-q * T) * pdf_d1 * sigma / (2 * math.sqrt(T))
            + r * K * math.exp(-r * T) * _norm_cdf(-d2)
            - q * S * math.exp(-q * T) * _norm_cdf(-d1)
        )
    return theta_annual / DAYS_PER_YEAR


# ---------------------------------------------------------------------------
# Strike-space Greeks (also simple closed forms -- low error risk)
# ---------------------------------------------------------------------------

def bs_dual_delta(S, K, T, r, q, sigma, option_type):
    """∂Price/∂K -- approximately the risk-neutral probability of finishing ITM."""
    if T <= 0 or sigma <= 0:
        return None
    _, d2 = _d1_d2(S, K, T, r, q, sigma)
    if option_type == "CE":
        return -math.exp(-r * T) * _norm_cdf(d2)
    return math.exp(-r * T) * _norm_cdf(-d2)


def bs_dual_gamma(S, K, T, r, q, sigma):
    """∂²Price/∂K² -- risk-neutral probability density at strike K."""
    if T <= 0 or sigma <= 0:
        return None
    _, d2 = _d1_d2(S, K, T, r, q, sigma)
    return math.exp(-r * T) * _norm_pdf(d2) / (K * sigma * math.sqrt(T))


def bs_omega(S, K, T, r, q, sigma, option_type):
    """
    Elasticity / leverage (a.k.a. lambda): %change in option price per
    %change in spot. Uses the THEORETICAL Black-Scholes price (not the raw
    market quote) as the denominator so it stays stable even when the
    market price is noisy or the option is far OTM with a tiny price.
    """
    if T <= 0 or sigma <= 0:
        return None
    price = bs_price(S, K, T, r, q, sigma, option_type)
    delta = bs_delta(S, K, T, r, q, sigma, option_type)
    if price is None or delta is None or price == 0:
        return None
    return delta * S / price


# ---------------------------------------------------------------------------
# Higher-order / cross Greeks -- central finite differences (see module
# docstring for why)
# ---------------------------------------------------------------------------

def compute_all_greeks(S, K, T, r, q, sigma, option_type):
    """
    Returns a dict with every Greek this pipeline stores. Returns a dict of
    Nones (all keys present, values None) if inputs are degenerate --
    caller decides how to handle that, matching the None-on-bad-input
    pattern used elsewhere in the pipeline (e.g. implied_vol()).
    """
    keys = [
        "delta", "gamma", "theta", "vega",
        "vanna", "charm", "vomma",
        "speed", "zomma", "color", "veta",
        "omega", "dual_delta", "dual_gamma",
    ]
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {k: None for k in keys}

    out = {
        "delta": bs_delta(S, K, T, r, q, sigma, option_type),
        "gamma": bs_gamma(S, K, T, r, q, sigma),
        "theta": bs_theta_per_day(S, K, T, r, q, sigma, option_type),
        "vega": bs_vega_per_1pct(S, K, T, r, q, sigma),
        "omega": bs_omega(S, K, T, r, q, sigma, option_type),
        "dual_delta": bs_dual_delta(S, K, T, r, q, sigma, option_type),
        "dual_gamma": bs_dual_gamma(S, K, T, r, q, sigma),
    }

    h_sigma = H_SIGMA
    h_S = max(S * H_S_PCT, 0.01)
    h_T = min(1.0 / (2 * DAYS_PER_YEAR), max(T / 20.0, 1e-6))

    # Vanna = ∂Delta/∂sigma
    d_up = bs_delta(S, K, T, r, q, sigma + h_sigma, option_type)
    d_dn = bs_delta(S, K, T, r, q, max(sigma - h_sigma, 1e-6), option_type)
    out["vanna"] = (d_up - d_dn) / (2 * h_sigma) if None not in (d_up, d_dn) else None

    # Vomma = ∂Vega/∂sigma
    v_up = bs_vega_per_1pct(S, K, T, r, q, sigma + h_sigma)
    v_dn = bs_vega_per_1pct(S, K, T, r, q, max(sigma - h_sigma, 1e-6))
    out["vomma"] = (v_up - v_dn) / (2 * h_sigma) if None not in (v_up, v_dn) else None

    # Speed = ∂Gamma/∂S
    g_up = bs_gamma(S + h_S, K, T, r, q, sigma)
    g_dn = bs_gamma(max(S - h_S, 1e-6), K, T, r, q, sigma)
    out["speed"] = (g_up - g_dn) / (2 * h_S) if None not in (g_up, g_dn) else None

    # Zomma = ∂Gamma/∂sigma
    gz_up = bs_gamma(S, K, T, r, q, sigma + h_sigma)
    gz_dn = bs_gamma(S, K, T, r, q, max(sigma - h_sigma, 1e-6))
    out["zomma"] = (gz_up - gz_dn) / (2 * h_sigma) if None not in (gz_up, gz_dn) else None

    # Color = ∂Gamma/∂t (per day). T decreases as calendar time passes, so
    # the "decay per day" sign is the negative of ∂Gamma/∂T.
    if T - h_T > 0:
        gc_up = bs_gamma(S, K, T + h_T, r, q, sigma)
        gc_dn = bs_gamma(S, K, T - h_T, r, q, sigma)
        dGamma_dT = (gc_up - gc_dn) / (2 * h_T) if None not in (gc_up, gc_dn) else None
        out["color"] = (-dGamma_dT / DAYS_PER_YEAR) if dGamma_dT is not None else None
    else:
        out["color"] = None

    # Charm = ∂Delta/∂t (per day), same sign logic as color
    if T - h_T > 0:
        dc_up = bs_delta(S, K, T + h_T, r, q, sigma, option_type)
        dc_dn = bs_delta(S, K, T - h_T, r, q, sigma, option_type)
        dDelta_dT = (dc_up - dc_dn) / (2 * h_T) if None not in (dc_up, dc_dn) else None
        out["charm"] = (-dDelta_dT / DAYS_PER_YEAR) if dDelta_dT is not None else None
    else:
        out["charm"] = None

    # Veta = ∂Vega/∂t (per day), same sign logic
    if T - h_T > 0:
        ve_up = bs_vega_per_1pct(S, K, T + h_T, r, q, sigma)
        ve_dn = bs_vega_per_1pct(S, K, T - h_T, r, q, sigma)
        dVega_dT = (ve_up - ve_dn) / (2 * h_T) if None not in (ve_up, ve_dn) else None
        out["veta"] = (-dVega_dT / DAYS_PER_YEAR) if dVega_dT is not None else None
    else:
        out["veta"] = None

    return out


# ---------------------------------------------------------------------------
# Implied volatility solver
# ---------------------------------------------------------------------------

def implied_vol(market_price, S, K, T, r, q, option_type, tol=1e-4, max_iter=100):
    """
    Newton-Raphson with a bisection fallback. Returns None if it fails to
    converge, or when market_price is below the no-arbitrage lower bound -- a
    strong sign of a bad/stale quote, in which case we don't force-fit an IV.
    """
    if T <= 0 or market_price <= 0 or S <= 0 or K <= 0:
        return None

    # The no-arbitrage floor for a EUROPEAN option is the DISCOUNTED intrinsic,
    # not max(S - K, 0) / max(K - S, 0). Those are the AMERICAN bounds (early
    # exercise is what makes the undiscounted payoff attainable today), and NSE
    # index options are European.
    #
    # Using the undiscounted bound rejects real quotes. For a deep-ITM put the
    # undiscounted K - S is the HIGHER of the two, so every legitimate price in
    # the band between K*e^-rT - S*e^-qT and K - S gets refused: at S=100,
    # K=115, T=1, r=0.05, q=0 the true floor is 9.391 and the BS price at
    # sigma=0.10 is 10.453, well under K - S = 15.
    forward = S * math.exp(-q * T)
    strike_pv = K * math.exp(-r * T)
    intrinsic = (max(forward - strike_pv, 0.0) if option_type == "CE"
                 else max(strike_pv - forward, 0.0))
    if market_price < intrinsic - 1e-6:
        return None

    sigma = 0.3
    for _ in range(max_iter):
        price = bs_price(S, K, T, r, q, sigma, option_type)
        d1, _ = _d1_d2(S, K, T, r, q, sigma)
        vega_raw = S * math.exp(-q * T) * _norm_pdf(d1) * math.sqrt(T)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        if vega_raw < 1e-8:
            break
        sigma -= diff / vega_raw
        if sigma <= 0:
            sigma = 0.01

    # Newton-Raphson failed to converge cleanly -- fall back to bisection
    lo, hi = 1e-4, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2
        price = bs_price(S, K, T, r, q, mid, option_type)
        if abs(price - market_price) < tol:
            return mid
        if price > market_price:
            hi = mid
        else:
            lo = mid
    return None
