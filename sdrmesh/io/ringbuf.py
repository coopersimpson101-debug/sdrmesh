"""Fixed-capacity ring buffer of complex samples with absolute sample indexing."""

from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    """Single-producer / single-consumer ring buffer.

    ``write`` appends samples (overwriting the oldest when full and counting
    the overrun); ``peek(n)`` returns the ``n`` oldest unread samples without
    consuming them; ``consume(n)`` drops them. ``read_index`` is the absolute
    index of the oldest unread sample, so decoders can time-stamp packets
    across blocks.
    """

    def __init__(self, capacity: int, dtype=np.complex64):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.buf = np.zeros(capacity, dtype=dtype)
        self.cap = capacity
        self.w = 0  # total samples ever written
        self.r = 0  # total samples ever consumed
        self.overruns = 0
        self.lock = threading.Lock()

    def __len__(self) -> int:
        return self.w - self.r

    @property
    def read_index(self) -> int:
        return self.r

    def write(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=self.buf.dtype)
        with self.lock:
            if len(x) >= self.cap:
                dropped = len(x) - self.cap
                x = x[-self.cap:]
                self.w += dropped
            n = len(x)
            i = self.w % self.cap
            k = min(n, self.cap - i)
            self.buf[i:i + k] = x[:k]
            self.buf[:n - k] = x[k:]
            self.w += n
            if self.w - self.r > self.cap:
                lost = self.w - self.r - self.cap
                self.overruns += lost
                self.r += lost

    def peek(self, n: int | None = None) -> np.ndarray:
        with self.lock:
            avail = self.w - self.r
            n = avail if n is None else min(n, avail)
            i = self.r % self.cap
            k = min(n, self.cap - i)
            return np.concatenate([self.buf[i:i + k], self.buf[:n - k]])

    def consume(self, n: int) -> None:
        with self.lock:
            self.r += min(n, self.w - self.r)
