"""Throughput of the streaming BLE receiver on a synthetic stream.

    python tools/bench_stream.py [--seconds 2] [--rate 4e6] [--repeat 3]

Builds ``seconds`` of complex baseband at ``rate`` with a meshfind advert
every 50 ms in AWGN (Eb/N0 16 dB), feeds it in 262 144-sample blocks (what
``hackrf_transfer`` delivers per USB transfer group is of that order) and
reports the decoder's busy time, the best of ``repeat`` runs.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sdrmesh import channel as ch  # noqa: E402
from sdrmesh.ble.gfsk import modulate_pdu  # noqa: E402
from sdrmesh.ble.linklayer import channel_freq  # noqa: E402
from sdrmesh.ble.pdu import adv_nonconn  # noqa: E402
from sdrmesh.io.stream import StreamingBleReceiver  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--rate", type=float, default=4e6)
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    n = int(a.seconds * a.rate)
    sps = int(a.rate / 1e6)
    x = np.zeros(n, dtype=complex)
    n_pk = 0
    for start in range(int(0.01 * a.rate), n - 2000 * sps, int(0.05 * a.rate)):
        s = modulate_pdu(adv_nonconn(bytes(rng.integers(0, 256, 6, dtype=np.uint8)),
                                     bytes(rng.integers(0, 256, 31, dtype=np.uint8))), 37, sps=sps)
        x[start:start + len(s)] = s
        n_pk += 1
    x = ch.awgn(x, ch.sigma2_for_ebn0(16, sps), rng)
    blocks = [x[i:i + 262144] for i in range(0, n, 262144)]
    best = None
    found = 0
    for _ in range(a.repeat):
        rx = StreamingBleReceiver(a.rate, channel_freq(37))
        t0 = time.perf_counter()
        found = sum(1 for _ in rx.run(blocks))
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    print(f"{platform.processor() or platform.machine()} | Python {platform.python_version()} | numpy {np.__version__}")
    print(f"{a.seconds:.1f} s of {a.rate / 1e6:g} MS/s with {n_pk} adverts: decoded {found}, "
          f"best wall time {best:.2f} s -> {a.seconds / best:.2f}x real time "
          f"({n / best / 1e6:.2f} MS/s sustained)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
