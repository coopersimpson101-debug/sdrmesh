"""Fractional-sample interpolation (cubic Lagrange / Farrow structure).

A cubic Lagrange interpolator through the four samples ``x[i-1..i+2]`` around a
fractional position ``t = i + mu`` (``0 <= mu < 1``) can be written in Farrow
form as a polynomial in ``mu`` whose coefficients are fixed FIR combinations of
the samples::

    y(mu) = ((c3 mu + c2) mu + c1) mu + c0
    c0 = x[i]
    c1 = -x[i-1]/3 - x[i]/2 + x[i+1] - x[i+2]/6
    c2 =  x[i-1]/2 - x[i]   + x[i+1]/2
    c3 = -x[i-1]/6 + x[i]/2 - x[i+1]/2 + x[i+2]/6

Reference: C. W. Farrow, "A continuously variable digital delay element",
ISCAS 1988; L. Erup, F. M. Gardner, R. A. Harris, "Interpolation in digital
modems, part II", IEEE Trans. Commun. 41(6), 1993.
"""

from __future__ import annotations

import numpy as np


def cubic_at(x: np.ndarray, t) -> np.ndarray:
    """Evaluate ``x`` at (possibly fractional) positions ``t`` (array or scalar).

    Positions outside ``[1, len(x) - 3]`` are clamped to the valid range.
    """
    x = np.asarray(x)
    t = np.asarray(t, dtype=float)
    n = len(x)
    if n < 4:
        raise ValueError("need at least 4 samples")
    t = np.clip(t, 1.0, n - 2.000001)
    i = np.floor(t).astype(np.int64)
    i = np.minimum(i, n - 3)
    mu = t - i
    xm1, x0, x1, x2 = x[i - 1], x[i], x[i + 1], x[i + 2]
    c0 = x0
    c1 = -xm1 / 3 - x0 / 2 + x1 - x2 / 6
    c2 = xm1 / 2 - x0 + x1 / 2
    c3 = -xm1 / 6 + x0 / 2 - x1 / 2 + x2 / 6
    return ((c3 * mu + c2) * mu + c1) * mu + c0


def fractional_delay(x: np.ndarray, delay: float) -> np.ndarray:
    """Delay ``x`` by ``delay`` samples (any real number >= 0), same length.

    The integer part is a shift (zeros enter at the start); the fractional part
    uses the cubic interpolator.
    """
    x = np.asarray(x)
    d_int = int(np.floor(delay))
    d_frac = delay - d_int
    pad = np.concatenate([np.zeros(d_int + 2, dtype=x.dtype), x, np.zeros(3, dtype=x.dtype)])
    pos = np.arange(len(x)) + 2 - d_frac
    return cubic_at(pad, pos)


def resample_ppm(x: np.ndarray, ppm: float) -> np.ndarray:
    """Model a sample-clock offset: the receiver clock runs ``ppm`` parts per
    million fast, so it takes samples at ``t_k = k (1 - ppm 1e-6)`` of the
    transmitter's sample grid."""
    x = np.asarray(x)
    step = 1.0 - ppm * 1e-6
    n_out = int((len(x) - 4) / step)
    pos = 1 + np.arange(n_out) * step
    return cubic_at(x, pos)


def fractional_delay_fft(x: np.ndarray, delay: float) -> np.ndarray:
    """Band-limited (ideal sinc) delay by ``delay`` samples, applied as a linear
    phase in the DFT domain. Exact for signals that are band-limited to the
    sampling bandwidth, which the cubic interpolator is not near fs/2; it is
    the right model for a critically sampled chirp. The signal is zero-padded
    so the circular shift does not wrap."""
    x = np.asarray(x, dtype=complex)
    pad = int(np.ceil(abs(delay))) + 16
    n = len(x) + 2 * pad
    nfft = 1 << (n - 1).bit_length()
    X = np.fft.fft(np.concatenate([np.zeros(pad), x]), nfft)
    k = np.fft.fftfreq(nfft)
    y = np.fft.ifft(X * np.exp(-2j * np.pi * k * delay))
    return y[pad:pad + len(x)]
