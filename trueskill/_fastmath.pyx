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


