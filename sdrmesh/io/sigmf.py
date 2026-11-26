"""Minimal SigMF (v1.0.0) metadata writer/reader.

A SigMF recording is a pair ``name.sigmf-data`` (raw samples) +
``name.sigmf-meta`` (JSON with ``global``, ``captures`` and ``annotations``).
Spec: https://github.com/sigmf/SigMF/blob/main/sigmf-spec.md
"""

from __future__ import annotations

import datetime as _dt
import json
import os

import numpy as np

from . import iq

SIGMF_VERSION = "1.0.0"


def base_path(path: str) -> str:
    for ext in (".sigmf-meta", ".sigmf-data"):
        if path.endswith(ext):
            return path[: -len(ext)]
    return path


def make_meta(fmt: str, sample_rate: float, center_freq: float, description: str = "",
              hw: str | None = None, annotations: list[dict] | None = None,
              datetime: str | None = None) -> dict:
    glob = {
        "core:datatype": iq.sigmf_datatype(fmt),
        "core:sample_rate": float(sample_rate),
        "core:version": SIGMF_VERSION,
        "core:recorder": "sdrmesh",
        "core:description": description,
    }
    if hw:
        glob["core:hw"] = hw
    cap = {"core:sample_start": 0, "core:frequency": float(center_freq)}
    if datetime:
        cap["core:datetime"] = datetime
    return {"global": glob, "captures": [cap], "annotations": annotations or []}


def annotation(sample_start: int, sample_count: int, label: str, f_lo: float | None = None,
               f_hi: float | None = None, comment: str | None = None) -> dict:
    a = {"core:sample_start": int(sample_start), "core:sample_count": int(sample_count), "core:label": label}
    if f_lo is not None and f_hi is not None:
        a["core:freq_lower_edge"] = float(f_lo)
        a["core:freq_upper_edge"] = float(f_hi)
    if comment:
        a["core:comment"] = comment
    return a


def write(path: str, x: np.ndarray, fmt: str, sample_rate: float, center_freq: float,
          description: str = "", annotations: list[dict] | None = None, hw: str | None = None,
          timestamp: bool = False) -> tuple[str, str]:
    """Write ``<base>.sigmf-data`` and ``<base>.sigmf-meta``. Returns both paths."""
    base = base_path(path)
    iq.write(base + ".sigmf-data", x, fmt)
    dt = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if timestamp else None
    meta = make_meta(fmt, sample_rate, center_freq, description, hw, annotations, dt)
    with open(base + ".sigmf-meta", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    return base + ".sigmf-data", base + ".sigmf-meta"


def write_meta_for(data_path: str, fmt: str, sample_rate: float, center_freq: float, **kw) -> str:
    """Write a ``.sigmf-meta`` next to an existing raw file (e.g. ``capture.cs8``)."""
    meta = make_meta(fmt, sample_rate, center_freq, **kw)
    meta["global"]["core:dataset"] = os.path.basename(data_path)
    out = os.path.splitext(data_path)[0] + ".sigmf-meta"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    return out


_DT_TO_FMT = {"ci8": "cs8", "ci16_le": "cs16", "cf32_le": "cf32"}


def read(path: str) -> tuple[np.ndarray, dict]:
    """Read a SigMF recording given either file of the pair."""
    base = base_path(path)
    with open(base + ".sigmf-meta", encoding="utf-8") as f:
        meta = json.load(f)
    dt = meta["global"]["core:datatype"]
    if dt not in _DT_TO_FMT:
        raise ValueError(f"unsupported SigMF datatype {dt}")
    data = base + ".sigmf-data"
    if not os.path.exists(data) and "core:dataset" in meta["global"]:
        data = os.path.join(os.path.dirname(base), meta["global"]["core:dataset"])
    return iq.read(data, _DT_TO_FMT[dt]), meta
