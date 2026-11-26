"""IQ sample file formats.

* ``cs8``  - interleaved signed 8-bit I/Q, what ``hackrf_transfer -r`` writes
  and ``-t`` reads (SigMF datatype ``ci8``).
* ``cs16`` - interleaved signed 16-bit little-endian I/Q (``ci16_le``; PlutoSDR,
  12-bit samples in 16-bit words).
* ``cf32`` - interleaved float32 little-endian I/Q (``cf32_le``; GNU Radio's
  ``gr_complex`` file sink).
"""

from __future__ import annotations

import numpy as np

FORMATS = {"cs8": ("ci8", np.int8, 127.0), "cs16": ("ci16_le", np.dtype("<i2"), 32767.0),
           "cf32": ("cf32_le", np.dtype("<f4"), 1.0)}


def sigmf_datatype(fmt: str) -> str:
    return FORMATS[fmt][0]


def from_bytes(raw: bytes, fmt: str = "cs8") -> np.ndarray:
    """Raw interleaved bytes -> complex128 scaled to about +-1 full scale.
    A trailing partial sample is ignored."""
    _, dt, scale = FORMATS[fmt]
    dt = np.dtype(dt)
    n = (len(raw) // (2 * dt.itemsize)) * 2
    a = np.frombuffer(raw, dtype=dt, count=n).astype(np.float64)
    return (a[0::2] + 1j * a[1::2]) / scale


def to_bytes(x: np.ndarray, fmt: str = "cs8", full_scale: float = 1.0) -> bytes:
    """Complex samples -> interleaved integers (clipped, rounded)."""
    _, dt, scale = FORMATS[fmt]
    x = np.asarray(x, dtype=complex) / full_scale
    inter = np.empty(2 * len(x))
    inter[0::2], inter[1::2] = x.real, x.imag
    if fmt == "cf32":
        return inter.astype(dt).tobytes()
    q = np.clip(np.round(inter * scale), -scale - 1, scale)
    return q.astype(dt).tobytes()


def read(path: str, fmt: str = "cs8", offset: int = 0, count: int | None = None) -> np.ndarray:
    """Read ``count`` samples starting at sample ``offset``."""
    _, dt, _ = FORMATS[fmt]
    item = 2 * np.dtype(dt).itemsize
    with open(path, "rb") as f:
        f.seek(offset * item)
        raw = f.read() if count is None else f.read(count * item)
    return from_bytes(raw, fmt)


def write(path: str, x: np.ndarray, fmt: str = "cs8", full_scale: float = 1.0) -> None:
    with open(path, "wb") as f:
        f.write(to_bytes(x, fmt, full_scale))


def fmt_from_path(path: str, default: str = "cs8") -> str:
    p = path.lower()
    for ext, fmt in ((".cs8", "cs8"), (".ci8", "cs8"), (".cs16", "cs16"), (".ci16", "cs16"),
                     (".cf32", "cf32"), (".fc32", "cf32"), (".cfile", "cf32")):
        if p.endswith(ext):
            return fmt
    return default
