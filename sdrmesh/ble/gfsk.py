"""GFSK modulator for the BLE LE 1M PHY (1 Msym/s, BT = 0.5, h = 0.5).

The complex baseband signal is ``s(t) = exp(j phi(t))`` with

    phi(t) = pi h  sum_k a_k  q(t - kT),     a_k = 2 b_k - 1,

where ``q`` is the integral of the frequency pulse ``g = rect_T * gauss_BT``
normalised so that ``q(inf) = 1``: every bit turns the phase by ``+-pi h``
(``+-pi/2`` for BLE). The instantaneous frequency deviation is
``+- h / (2T) = +-250 kHz``. The Bluetooth spec allows h in 0.45..0.55.

Reference: Bluetooth Core Spec v5.4 Vol 6 Part A 3.1; J. G. Proakis,
*Digital Communications*, 5th ed., 3.3 (CPM).
"""

from __future__ import annotations

import math

import numpy as np

from ..dsp.filters import filt, gaussian_pulse
from .linklayer import build_air_bits
from .pdu import AdvPdu

SYMBOL_RATE = 1e6


def frequency_waveform(bits: np.ndarray, sps: int, bt: float = 0.5) -> np.ndarray:
    """Normalised instantaneous frequency (+-1 for long runs), ``len(bits)*sps``
    samples, with bit ``k``'s pulse centred on sample ``(k + 0.5) sps``."""
    nrz = 2.0 * np.asarray(bits, dtype=float) - 1.0
    rect = np.repeat(nrz, sps)
    g = gaussian_pulse(bt, sps, span=3)
    # pad with the edge symbols so the filter does not ramp to zero frequency
    pad = len(g)
    ext = np.concatenate([np.full(pad, rect[0]), rect, np.full(pad, rect[-1])])
    return filt(ext, g)[pad:pad + len(rect)]


def modulate_bits(bits: np.ndarray, sps: int = 4, bt: float = 0.5, h: float = 0.5,
                  guard_symbols: int = 8, phase0: float = 0.0, ramp_symbols: int = 2) -> np.ndarray:
    """Bits -> complex baseband, unit amplitude.

    ``ramp_symbols`` of unmodulated carrier (phase-continuous) precede and
    follow the packet, as a real transmitter's PA ramp-up/down does, so the
    first and last bits see a carrier on both sides; ``guard_symbols`` of
    silence (zeros) are added outside that, so packets can be concatenated."""
    f = frequency_waveform(bits, sps, bt)
    r = np.zeros(ramp_symbols * sps)
    f = np.concatenate([r, f, r])
    phi = phase0 + math.pi * h * np.cumsum(f) / sps
    s = np.exp(1j * phi)
    z = np.zeros(guard_symbols * sps, dtype=complex)
    return np.concatenate([z, s, z])


def modulate_pdu(pdu: AdvPdu | bytes, channel: int = 37, sps: int = 4, **kw) -> np.ndarray:
    """Encode, CRC, whiten and modulate one advertising packet."""
    raw = pdu.encode() if isinstance(pdu, AdvPdu) else bytes(pdu)
    return modulate_bits(build_air_bits(raw, channel), sps=sps, **kw)
