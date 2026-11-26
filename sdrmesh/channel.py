"""Channel model: AWGN, carrier frequency offset, timing offset, sample-clock
offset and static/random multipath.

Noise calibration (complex baseband, sample rate ``fs``, unit-power signal)
---------------------------------------------------------------------------
Complex white noise of total variance ``sigma^2`` per sample has a two-sided
PSD ``N0 = sigma^2 / fs``. For a signal of power ``P``:

* in-band SNR over a bandwidth ``B``:  ``SNR = P / (N0 B) = P fs / (sigma^2 B)``
* ``Es/N0 = P T_s / N0 = P (fs T_s) / sigma^2 = P * samples_per_symbol / sigma^2``
* ``Eb/N0 = Es/N0 / bits_per_symbol``

so ``sigma^2 = P * sps / (bits_per_symbol * Eb/N0)``.
"""

from __future__ import annotations

import math

import numpy as np

from .dsp.filters import freq_shift
from .dsp.interp import fractional_delay, fractional_delay_fft, resample_ppm


def db2lin(db: float) -> float:
    return 10.0 ** (db / 10.0)


def awgn(x: np.ndarray, sigma2: float, rng: np.random.Generator) -> np.ndarray:
    n = rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x))
    return x + math.sqrt(sigma2 / 2) * n


def sigma2_for_ebn0(ebn0_db: float, sps: float, bits_per_symbol: float = 1.0, power: float = 1.0) -> float:
    return power * sps / (bits_per_symbol * db2lin(ebn0_db))


def sigma2_for_snr(snr_db: float, fs: float, bw: float, power: float = 1.0) -> float:
    """Noise variance per sample at rate ``fs`` for an in-band SNR over ``bw``."""
    return power * fs / (bw * db2lin(snr_db))


def multipath(x: np.ndarray, delays: list[float], gains: list[complex], bandlimited: bool = True) -> np.ndarray:
    """Tapped-delay-line channel ``y = sum_k g_k x(t - d_k)``; ``delays`` in
    samples (fractional allowed). Gains are used as given (normalise them
    yourself if unit average power is wanted)."""
    y = np.zeros(len(x), dtype=complex)
    for d, g in zip(delays, gains):
        if d == 0:
            y += g * x
        else:
            y += g * (fractional_delay_fft(x, d) if bandlimited else fractional_delay(x, d))
    return y


def rayleigh_taps(n_taps: int, spacing: float, decay_db_per_tap: float,
                  rng: np.random.Generator) -> tuple[list[float], list[complex]]:
    """Random taps with an exponential power-delay profile, unit total mean power."""
    p = np.array([db2lin(-decay_db_per_tap * k) for k in range(n_taps)])
    p /= p.sum()
    g = np.sqrt(p / 2) * (rng.standard_normal(n_taps) + 1j * rng.standard_normal(n_taps))
    return [k * spacing for k in range(n_taps)], list(g)


def apply(x: np.ndarray, fs: float, rng: np.random.Generator, *, cfo_hz: float = 0.0,
          delay: float = 0.0, ppm: float = 0.0, phase: float | None = None,
          taps: tuple[list[float], list[complex]] | None = None, sigma2: float = 0.0,
          lead: int = 0, tail: int = 0) -> np.ndarray:
    """Full impairment chain, in physical order: multipath -> timing (fractional
    delay, clock offset) -> CFO and carrier phase -> AWGN. ``lead``/``tail``
    zeros are added before the noise so a receiver must search for the packet."""
    y = np.asarray(x, dtype=complex)
    if taps is not None:
        y = multipath(y, *taps)
    if delay:
        y = fractional_delay_fft(y, delay)
    if ppm:
        y = resample_ppm(y, ppm)
    if phase is None:
        phase = float(rng.uniform(0, 2 * math.pi))
    y = freq_shift(y, cfo_hz, fs) * np.exp(1j * phase)
    if lead or tail:
        y = np.concatenate([np.zeros(lead, dtype=complex), y, np.zeros(tail, dtype=complex)])
    if sigma2 > 0:
        y = awgn(y, sigma2, rng)
    return y
