"""FIR filter design and channel selection, written from first principles.

NumPy is used only for array arithmetic and ``np.convolve``/FFTs; the filter
designs (windowed sinc, Gaussian pulse) are derived here.

References
----------
* A. V. Oppenheim, R. W. Schafer, *Discrete-Time Signal Processing*, 3rd ed.,
  section 7.5 (window method).
* F. J. Harris, "On the use of windows for harmonic analysis with the discrete
  Fourier transform", Proc. IEEE 66(1), 1978.
"""

from __future__ import annotations

import math

import numpy as np


def window(name: str, n: int) -> np.ndarray:
    """Symmetric window of length ``n`` (Hann, Hamming or Blackman)."""
    if n < 1:
        raise ValueError("window length must be >= 1")
    if n == 1:
        return np.ones(1)
    k = np.arange(n)
    x = 2.0 * math.pi * k / (n - 1)
    if name == "hann":
        return 0.5 - 0.5 * np.cos(x)
    if name == "hamming":
        return 0.54 - 0.46 * np.cos(x)
    if name == "blackman":
        return 0.42 - 0.5 * np.cos(x) + 0.08 * np.cos(2 * x)
    raise ValueError(f"unknown window {name!r}")


def lowpass(num_taps: int, cutoff: float, fs: float, win: str = "blackman") -> np.ndarray:
    """Linear-phase windowed-sinc low-pass FIR.

    ``cutoff`` is the -6 dB edge in Hz. Taps are normalised to unity DC gain.
    ``num_taps`` should be odd so the group delay is an integer number of
    samples, ``(num_taps - 1) / 2``.
    """
    if not 0 < cutoff < fs / 2:
        raise ValueError("cutoff must be in (0, fs/2)")
    m = np.arange(num_taps) - (num_taps - 1) / 2.0
    fc = cutoff / fs  # cycles / sample
    h = 2 * fc * np.sinc(2 * fc * m)  # ideal brick-wall impulse response
    h *= window(win, num_taps)
    return h / h.sum()


def gaussian_pulse(bt: float, sps: int, span: int = 3) -> np.ndarray:
    """Gaussian frequency-shaping pulse for GFSK, unit area.

    The Gaussian filter has impulse response
    ``h(t) = sqrt(2 pi / ln 2) B exp(-2 pi^2 B^2 t^2 / ln 2)`` with ``B = BT / T``.
    Sampled at ``sps`` samples per symbol over ``span`` symbols.
    """
    n = span * sps + 1
    t = (np.arange(n) - (n - 1) / 2.0) / sps  # in symbol periods
    b = bt
    h = math.sqrt(2 * math.pi / math.log(2)) * b * np.exp(-2 * math.pi ** 2 * b ** 2 * t ** 2 / math.log(2))
    return h / h.sum()


def filt(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Zero-phase-aligned FIR filtering: output has the same length as ``x``
    and the linear-phase group delay is removed."""
    if len(h) > 64 and len(x) > 4 * len(h):
        n = len(x) + len(h) - 1
        nfft = 1 << (n - 1).bit_length()
        y = np.fft.ifft(np.fft.fft(x, nfft) * np.fft.fft(h, nfft))[:n]
        if not (np.iscomplexobj(x) or np.iscomplexobj(h)):
            y = y.real
    else:
        y = np.convolve(x, h, mode="full")
    d = (len(h) - 1) // 2
    return y[d:d + len(x)]


def freq_shift(x: np.ndarray, f: float, fs: float, n0: int = 0) -> np.ndarray:
    """Multiply by ``exp(j 2 pi f n / fs)`` (an NCO). ``n0`` keeps phase continuous
    across streaming blocks."""
    n = np.arange(n0, n0 + len(x))
    return x * np.exp(2j * math.pi * f / fs * n)


def channel_select(x: np.ndarray, fs: float, offset: float, out_fs: float,
                   bandwidth: float, num_taps: int | None = None) -> np.ndarray:
    """Pick one channel out of a wider capture.

    1. mix the channel at ``offset`` Hz (relative to the capture centre) to DC,
    2. low-pass to ``bandwidth / 2``,
    3. decimate by the integer ``fs / out_fs``.
    """
    dec = fs / out_fs
    if abs(dec - round(dec)) > 1e-9 or dec < 1:
        raise ValueError("fs / out_fs must be a positive integer")
    dec = int(round(dec))
    y = freq_shift(x, -offset, fs) if offset else np.asarray(x, dtype=complex)
    if dec > 1 or bandwidth < fs:
        if num_taps is None:
            num_taps = 8 * dec + 1 if dec > 1 else 33
            num_taps |= 1
        h = lowpass(num_taps, min(bandwidth / 2, 0.49 * fs), fs)
        y = filt(y, h)
    return y[::dec]


def decimate(x: np.ndarray, factor: int, num_taps: int | None = None) -> np.ndarray:
    """Anti-aliased integer decimation."""
    if factor == 1:
        return np.asarray(x)
    if num_taps is None:
        num_taps = 10 * factor + 1
    h = lowpass(num_taps | 1, 0.45 / factor, 1.0)
    return filt(x, h)[::factor]
