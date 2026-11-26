"""Short-time Fourier transform / spectrogram (own framing; ``np.fft`` for the DFT)."""

from __future__ import annotations

import numpy as np

from .filters import window


def spectrogram(x: np.ndarray, fs: float, nfft: int = 256, hop: int | None = None,
                win: str = "hann") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Power spectrogram of a complex signal.

    Returns ``(t, f, S_db)`` where ``S_db[i, k]`` is the power in frame ``i`` and
    frequency bin ``k`` (fft-shifted, so ``f`` runs from ``-fs/2`` to ``fs/2``),
    normalised so that a unit-power tone reads 0 dB.
    """
    x = np.asarray(x, dtype=complex)
    if hop is None:
        hop = nfft // 4
    w = window(win, nfft)
    n_frames = max(1, 1 + (len(x) - nfft) // hop)
    idx = np.arange(nfft)[None, :] + hop * np.arange(n_frames)[:, None]
    idx = np.minimum(idx, len(x) - 1)
    frames = x[idx] * w[None, :]
    X = np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)
    p = np.abs(X) ** 2 / (w.sum() ** 2)
    s_db = 10 * np.log10(p + 1e-20)
    t = (hop * np.arange(n_frames) + nfft / 2) / fs
    f = (np.arange(nfft) - nfft // 2) * fs / nfft
    return t, f, s_db


def psd_welch(x: np.ndarray, fs: float, nfft: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Averaged periodogram (Welch, 50 % overlap, Hann), in power per bin (linear)."""
    _, f, s_db = spectrogram(x, fs, nfft=nfft, hop=nfft // 2)
    return f, np.mean(10 ** (s_db / 10), axis=0)
