"""Gardner symbol-timing recovery with a cubic (Farrow) interpolator.

The Gardner timing-error detector works at two samples per symbol on the
interpolated signal ``y``::

    e_k = y(t_k - T/2) * (y(t_k) - y(t_{k-1}))

At a data transition the mid-point sample sits on the zero crossing when the
strobes ``t_k`` are on the symbol centres; a late strobe moves the mid-point
past the crossing and makes ``e_k`` take the sign of the transition, so the
loop updates

    t_{k+1} = t_k + T_k - Kp e_k,      T_{k+1} = T_k - Ki e_k

(a proportional-integral loop filter: the integral branch tracks a sample-clock
frequency offset). It needs no decisions and is insensitive to carrier phase,
which suits the FM-discriminator output used here.

Reference: F. M. Gardner, "A BPSK/QPSK timing-error detector for sampled
receivers", IEEE Trans. Commun. 34(5), 1986; M. Rice, *Digital
Communications: A Discrete-Time Approach*, ch. 8.
"""

from __future__ import annotations

import math


def _cubic(x, t: float) -> float:
    """Scalar cubic Lagrange interpolation (pure Python is faster than NumPy
    for one point)."""
    n = len(x)
    if t < 1.0:
        t = 1.0
    if t > n - 2.000001:
        t = n - 2.000001
    i = int(math.floor(t))
    if i > n - 3:
        i = n - 3
    mu = t - i
    xm1, x0, x1, x2 = float(x[i - 1]), float(x[i]), float(x[i + 1]), float(x[i + 2])
    c1 = -xm1 / 3 - x0 / 2 + x1 - x2 / 6
    c2 = xm1 / 2 - x0 + x1 / 2
    c3 = -xm1 / 6 + x0 / 2 - x1 / 2 + x2 / 6
    return ((c3 * mu + c2) * mu + c1) * mu + x0


class GardnerTiming:
    """Stateful timing loop over a real-valued signal ``y``.

    ``t0`` is the (fractional) sample index of the first symbol centre.
    """

    def __init__(self, y, t0: float, sps: float, kp: float = 0.12, ki: float = 0.0015):
        self.list_y = y.tolist() if hasattr(y, "tolist") else list(y)
        self.t = float(t0)
        self.T = float(sps)
        self.sps = float(sps)
        self.kp = kp * sps / 4.0
        self.ki = ki * sps / 4.0
        self.prev = _cubic(self.list_y, self.t - self.T)
        self.strobes: list[float] = []

    def remaining(self) -> int:
        return int((len(self.list_y) - 3 - self.t) // self.T)

    def run(self, n: int) -> list[float]:
        """Produce ``n`` more soft symbols (fewer if the signal ends)."""
        y = self.list_y
        out = []
        lim = len(y) - 3
        for _ in range(n):
            if self.t >= lim:
                break
            yk = _cubic(y, self.t)
            ym = _cubic(y, self.t - self.T / 2)
            e = ym * (yk - self.prev)
            # clamp the error: a single noisy transition must not throw the loop
            if e > 1.0:
                e = 1.0
            elif e < -1.0:
                e = -1.0
            out.append(yk)
            self.strobes.append(self.t)
            self.prev = yk
            self.T -= self.ki * e
            # keep the period within +-2 % of nominal (BLE allows +-50 ppm)
            lo, hi = 0.98 * self.sps, 1.02 * self.sps
            self.T = lo if self.T < lo else hi if self.T > hi else self.T
            self.t += self.T - self.kp * e
        return out
