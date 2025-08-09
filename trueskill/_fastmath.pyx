# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""
Fast math helpers for TrueSkill using Cython.

Provides erfc, cdf, pdf, ppf compatible with backends API.
"""
cdef extern from "math.h":
    double erf(double)
    double erfc(double)
    double sqrt(double)
    double exp(double)
    double fabs(double)
    double log(double)
    double M_PI

cdef inline double _phi(double x, double mu, double sigma):
    # PDF of normal distribution
    cdef double s = fabs(sigma)
    return (1.0 / (sqrt(2.0 * M_PI) * s)) * exp(-(((x - mu) / s) * ((x - mu) / s)) / 2.0)

cpdef double erfc_py(double x):
    return erfc(x)

cpdef double cdf(double x, double mu=0.0, double sigma=1.0):
    # 0.5 * erfc(-(x-mu)/(sigma*sqrt(2)))
    return 0.5 * erfc(- (x - mu) / (sigma * sqrt(2.0)))

cpdef double pdf(double x, double mu=0.0, double sigma=1.0):
    return _phi(x, mu, sigma)

cdef inline double _erfcinv(double y):
    # Approximate erfcinv using the same algorithm as Python backend
    cdef double t, x, err
    cdef bint zero_point
    if y >= 2.0:
        return -100.0
    elif y <= 0.0:
        return 100.0
    zero_point = y < 1.0
    if not zero_point:
        y = 2.0 - y
    t = sqrt(-2.0 * log(y / 2.0))
    x = -0.70711 * ((2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t)
    for _ in range(2):
        err = erfc(x) - y
        x += err / (1.12837916709551257 * exp(-(x * x)) - x * err)
    return x if zero_point else -x

cpdef double ppf(double x, double mu=0.0, double sigma=1.0):
    # Inverse CDF using erfcinv approximation
    return mu - sigma * sqrt(2.0) * _erfcinv(2.0 * x)


# ----- V/W functions (standardized) -----
cdef inline double _v_win(double diff, double draw_margin):
    cdef double x = diff - draw_margin
    cdef double denom = cdf(x)
    if denom != 0.0:
        return pdf(x) / denom
    return -x

cdef inline double _w_win(double diff, double draw_margin):
    cdef double x = diff - draw_margin
    cdef double v = _v_win(diff, draw_margin)
    cdef double w = v * (v + x)
    # Caller is responsible for checking pathological cases
    return w

cdef inline double _v_draw(double diff, double draw_margin):
    cdef double abs_diff = diff if diff >= 0 else -diff
    cdef double a = draw_margin - abs_diff
    cdef double b = -draw_margin - abs_diff
    cdef double denom = cdf(a) - cdf(b)
    cdef double numer = pdf(b) - pdf(a)
    cdef double val
    if denom != 0.0:
        val = numer / denom
    else:
        val = a
    return -val if diff < 0 else val

cdef inline double _w_draw(double diff, double draw_margin):
    cdef double abs_diff = diff if diff >= 0 else -diff
    cdef double a = draw_margin - abs_diff
    cdef double b = -draw_margin - abs_diff
    cdef double denom = cdf(a) - cdf(b)
    if denom == 0.0:
        # Caller handles error path
        return 0.0
    cdef double v = _v_draw(abs_diff, draw_margin)
    return (v * v) + (a * pdf(a) - b * pdf(b)) / denom


cpdef tuple truncate_up(double div_pi, double div_tau, double draw_margin, bint is_draw):
    """Fast numeric core for TruncateFactor.up.

    Returns (pi, tau) for the updated variable value.
    """
    cdef double sqrt_pi = sqrt(div_pi)
    cdef double diff = div_tau / sqrt_pi
    cdef double margin = draw_margin * sqrt_pi
    cdef double v, w
    if is_draw:
        v = _v_draw(diff, margin)
        w = _w_draw(diff, margin)
    else:
        v = _v_win(diff, margin)
        w = _w_win(diff, margin)
    cdef double denom = 1.0 - w
    cdef double pi = div_pi / denom
    cdef double tau = (div_tau + sqrt_pi * v) / denom
    return pi, tau


cpdef tuple sum_update(div_pi, div_tau, coeffs):
    """Fast numeric core for SumFactor.update.

    Given arrays of div.pi and div.tau and coefficients, compute (pi, tau).
    """
    cdef Py_ssize_t n = len(div_pi)
    cdef double pi_inv = 0.0
    cdef double mu = 0.0
    cdef double d_pi, d_tau, c, mu_i
    for i in range(n):
        d_pi = <double>div_pi[i]
        d_tau = <double>div_tau[i]
        c = <double>coeffs[i]
        mu_i = d_tau / d_pi
        mu += c * mu_i
        pi_inv += (c * c) / d_pi
    cdef double pi = 1.0 / pi_inv
    cdef double tau = pi * mu
    return pi, tau


cpdef tuple rate_1vs1_fast(double mu1, double sigma1, double mu2, double sigma2,
                           double beta, double tau, double draw_probability,
                           bint drawn):
    """Closed-form 1v1 update matching TrueSkill with dynamic factor and draw.

    Returns (mu1', sigma1', mu2', sigma2').
    """
    # Inflate with dynamic factor
    cdef double s1_2 = sigma1 * sigma1 + tau * tau
    cdef double s2_2 = sigma2 * sigma2 + tau * tau
    cdef double c2 = s1_2 + s2_2 + 2.0 * beta * beta
    cdef double cstd = sqrt(c2)
    # Draw margin for two players
    cdef double margin = ppf((draw_probability + 1.0) / 2.0) * sqrt(2.0) * beta
    cdef double t = (mu1 - mu2) / cstd
    cdef double m = margin / cstd
    cdef double v, w
    if drawn:
        v = _v_draw(t, m)
        w = _w_draw(t, m)
    else:
        v = _v_win(t, m)
        w = _w_win(t, m)
    # Means
    cdef double mu1p = mu1 + (s1_2 / cstd) * v
    cdef double mu2p = mu2 - (s2_2 / cstd) * v
    # Variances
    cdef double s1p_2 = s1_2 * (1.0 - (s1_2 / c2) * w)
    cdef double s2p_2 = s2_2 * (1.0 - (s2_2 / c2) * w)
    if s1p_2 <= 0.0:
        s1p_2 = 1e-12
    if s2p_2 <= 0.0:
        s2p_2 = 1e-12
    return mu1p, sqrt(s1p_2), mu2p, sqrt(s2p_2)


