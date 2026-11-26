"""Analytic reference curves.

BLE / binary FSK (Proakis, *Digital Communications*, 5th ed., 4.5)
------------------------------------------------------------------
* non-coherent orthogonal BFSK:  ``P_b = 1/2 exp(-Eb / 2N0)``
* coherent MSK / BPSK (lower bound for h = 0.5 CPFSK): ``P_b = Q(sqrt(2 Eb/N0))``

BLE uses h = 0.5, which is *not* orthogonal for non-coherent detection, and a
limiter-discriminator receiver; the two curves bracket what a practical
discriminator receiver achieves.

LoRa (uncoded, ideal synchronisation)
-------------------------------------
After dechirping, the ``N = 2^SF`` symbols are orthogonal tones, so the
symbol error probability is that of non-coherent orthogonal N-FSK. With the
correct bin's envelope Rician (``a^2 = 2 Es/N0``) and the other ``N - 1``
Rayleigh, all normalised to unit noise variance per dimension::

    P_c = int_0^inf  r exp(-(r^2 + a^2)/2) I0(a r) (1 - exp(-r^2/2))^(N-1) dr
    P_s = 1 - P_c,       P_b = P_s * (N/2) / (N - 1)

and ``Es/N0 = N * SNR`` for a critically sampled chirp (SNR over BW). The
alternating-binomial closed form is numerically useless for N up to 4096, so
the integral is evaluated directly (log domain, trapezoid rule). See also
Reynders & Pollin, "Chirp spread spectrum as a modulation technique for long
range communication", SCVT 2016, and Elshabrawy & Robert, IEEE Commun. Lett. 22(9), 2018.
"""

from __future__ import annotations

import math

import numpy as np


def qfunc(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * np.vectorize(math.erfc)(x / math.sqrt(2))


def ber_bfsk_noncoherent(ebn0_db):
    g = 10 ** (np.asarray(ebn0_db, dtype=float) / 10)
    return 0.5 * np.exp(-g / 2)


def ber_msk_coherent(ebn0_db):
    g = 10 ** (np.asarray(ebn0_db, dtype=float) / 10)
    return qfunc(np.sqrt(2 * g))


def _log_i0e(x: np.ndarray) -> np.ndarray:
    """log(I0(x) e^{-x}) for x >= 0, without overflow."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = x < 50
    out[small] = np.log(np.i0(x[small])) - x[small]
    xl = x[~small]
    # asymptotic series I0(x) ~ e^x / sqrt(2 pi x) (1 + 1/(8x) + 9/(128 x^2) + 225/(3072 x^3))
    out[~small] = -0.5 * np.log(2 * math.pi * xl) + np.log1p(1 / (8 * xl) + 9 / (128 * xl ** 2) + 225 / (3072 * xl ** 3))
    return out


def ser_mfsk_noncoherent(n: int, esn0_lin: float, grid: int = 6000) -> float:
    """Symbol error probability of non-coherent orthogonal n-FSK."""
    a = math.sqrt(2 * esn0_lin)
    r_max = a + 12 + math.sqrt(2 * math.log(max(n, 2))) * 2
    r = np.linspace(1e-9, r_max, grid)
    # log of the integrand: log r - (r - a)^2 / 2 + log(I0(ar) e^{-ar}) + (n-1) log(1 - e^{-r^2/2})
    log_f = (np.log(r) - (r - a) ** 2 / 2 + _log_i0e(a * r)
             + (n - 1) * np.log(-np.expm1(-r * r / 2)))
    f = np.exp(log_f)
    pc = float(np.sum((f[1:] + f[:-1]) * np.diff(r)) / 2)
    return min(max(1.0 - pc, 0.0), 1.0)


def lora_ser(sf: int, snr_db) -> np.ndarray:
    n = 1 << sf
    snr = 10 ** (np.atleast_1d(np.asarray(snr_db, dtype=float)) / 10)
    return np.array([ser_mfsk_noncoherent(n, n * s) for s in snr])


def lora_ber(sf: int, snr_db) -> np.ndarray:
    n = 1 << sf
    return lora_ser(sf, snr_db) * (n / 2) / (n - 1)


def lora_snr_to_ebn0(sf: int, snr_db, cr: int | None = None):
    """Eb/N0 = SNR * 2^SF / SF (uncoded) or ``/ (SF * 4/(4+CR))`` (coded)."""
    rate = sf if cr is None else sf * 4 / (4 + cr)
    return np.asarray(snr_db, dtype=float) + 10 * np.log10((1 << sf) / rate)
