"""Chirp spread spectrum (CSS) waveforms.

A LoRa symbol of value ``s`` in ``0..N-1`` (``N = 2^SF``) is a linear
up-chirp cyclically shifted by ``s`` chips. At ``os`` samples per chip
(``fs = os BW``), sample ``n = 0..N os - 1``::

    x_s[n] = exp(j 2 pi ( n^2 / (2 N os^2) + (s / N - 1/2) n / os ))     for n < (N - s) os
    x_s[n] = exp(j 2 pi ( n^2 / (2 N os^2) + (s / N - 3/2) n / os ))     afterwards

i.e. the instantaneous frequency starts at ``(s/N - 1/2) BW``, rises at
``BW^2 / 2^SF`` Hz/s and wraps from ``+BW/2`` to ``-BW/2``.

**Demodulation (dechirp + FFT).** Multiplying by the conjugate base chirp
``x_0*[n]`` leaves a tone at ``s`` cycles per symbol, so at ``os = 1`` the
symbol is ``argmax_k |DFT_N{x[n] x_0*[n]}[k]|``. The N symbols are orthogonal,
so with ideal sync LoRa is non-coherent orthogonal N-ary signalling.

References: A. J. Berni, W. D. Gregg, "On the utility of chirp modulation for
digital signaling", IEEE Trans. Commun. 21(6), 1973; L. Vangelista, "Frequency
shift chirp modulation: the LoRa modulation", IEEE Signal Process. Lett.
24(12), 2017.
"""

from __future__ import annotations

import math

import numpy as np


def upchirp(sf: int, s: int = 0, os: int = 1) -> np.ndarray:
    N = 1 << sf
    n = np.arange(N * os)
    fold = (N - s) * os
    f0 = np.where(n < fold, s / N - 0.5, s / N - 1.5)
    return np.exp(2j * math.pi * (n * n / (2 * N * os * os) + f0 * n / os))


def downchirp(sf: int, os: int = 1) -> np.ndarray:
    return np.conj(upchirp(sf, 0, os))


def modulate_symbols(symbols, sf: int, os: int = 1) -> np.ndarray:
    """Concatenate chirps for a list of symbol values (vectorised)."""
    symbols = np.asarray(symbols, dtype=np.int64)
    N = 1 << sf
    n = np.arange(N * os)[None, :]
    s = symbols[:, None]
    fold = (N - s) * os
    f0 = np.where(n < fold, s / N - 0.5, s / N - 1.5)
    return np.exp(2j * math.pi * (n * n / (2 * N * os * os) + f0 * n / os)).ravel()


def dechirp_fft(x: np.ndarray, sf: int, ref: np.ndarray | None = None) -> np.ndarray:
    """``|DFT|`` of each N-sample symbol of ``x`` (critically sampled) after
    multiplying by ``ref`` (default: the down-chirp). Shape (n_symbols, N)."""
    N = 1 << sf
    if ref is None:
        ref = downchirp(sf)
    m = len(x) // N
    X = np.fft.fft(np.asarray(x[:m * N]).reshape(m, N) * ref[None, :], axis=1)
    return X


def demod_symbols(x: np.ndarray, sf: int) -> np.ndarray:
    """Hard symbol decisions (argmax of the dechirped spectrum)."""
    return np.argmax(np.abs(dechirp_fft(x, sf)), axis=1)
