"""Streaming receive pipeline.

    source blocks --> StreamChannelizer (NCO + FIR + decimate, stateful)
                  --> RingBuffer (channel-rate samples, absolute indices)
                  --> BleReceiver on overlapping windows
                  --> de-duplicated packets

The decoder looks at ``block + overlap`` samples but only accepts packets
whose sync word starts in the first ``block``; ``overlap`` is one maximum-
length packet, so every packet is decoded exactly once whatever the block
boundaries.
"""

from __future__ import annotations

import math
import time
from typing import Iterable, Iterator

import numpy as np

from ..ble.linklayer import channel_freq
from ..ble.pdu import BlePacket
from ..ble.receiver import SYNC_LEN, BleReceiver
from ..dsp.filters import lowpass
from .ringbuf import RingBuffer


class StreamChannelizer:
    """Block-wise channel selection identical to filtering the whole stream:
    keeps NCO phase, FIR history and decimation phase across blocks."""

    def __init__(self, fs_in: float, offset: float, fs_out: float, bandwidth: float = 2e6):
        dec = fs_in / fs_out
        if abs(dec - round(dec)) > 1e-9 or dec < 1:
            raise ValueError("input rate must be an integer multiple of the output rate")
        self.dec = int(round(dec))
        if abs(offset) + bandwidth / 2 > fs_in / 2:
            raise ValueError("channel is outside the captured bandwidth")
        self.fs_in, self.offset = fs_in, offset
        self.h = lowpass(8 * self.dec + 1, min(bandwidth / 2, 0.45 * fs_out), fs_in) if self.dec > 1 else None
        self.n = 0  # input samples consumed (NCO phase)
        self.hist = np.zeros(0 if self.h is None else len(self.h) - 1, dtype=complex)
        self.phase = 0  # decimation phase carried to the next block

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=complex)
        if self.offset:
            k = np.arange(self.n, self.n + len(x))
            x = x * np.exp(-2j * math.pi * self.offset / self.fs_in * k)
        self.n += len(x)
        if self.h is None:
            return x
        z = np.concatenate([self.hist, x])
        y = np.convolve(z, self.h, mode="valid")  # one output per new input sample
        self.hist = z[len(z) - (len(self.h) - 1):]
        out = y[self.phase::self.dec]
        self.phase = (self.phase - len(y)) % self.dec
        return out

    @property
    def delay_out(self) -> float:
        """Group delay of the channel filter in output samples."""
        return 0.0 if self.h is None else (len(self.h) - 1) / 2 / self.dec


class StreamingBleReceiver:
    """``max_pdu`` defaults to 39 octets: the longest PDU allowed on the
    primary advertising channels (legacy adverts, CONNECT_IND); this keeps
    the block overlap short."""

    def __init__(self, fs_in: float, center_freq: float, channel: int = 37, fs_out: float | None = None,
                 block: int | None = None, max_pdu: int = 39, **rx_kw):
        self.channel = channel
        if fs_out is None:
            # 2 MS/s (2 samples/symbol): measured here to be both ~1.5 dB better in BER and about
            # twice as fast as 4 MS/s with this front end (see README, Results)
            fs_out = 2e6
        offset = channel_freq(channel) - center_freq
        self.chan = StreamChannelizer(fs_in, offset, fs_out)
        self.rx = BleReceiver(fs_out, channel, max_pdu=max_pdu, **rx_kw)
        self.overlap = (SYNC_LEN + 8 * (2 + max_pdu + 3) + 32) * self.rx.sps
        if block is None:
            # largest block whose correlation FFT still fits in 2^16 points
            block = (1 << 16) - self.overlap - SYNC_LEN * self.rx.sps
        self.block = block
        self.ring = RingBuffer(4 * (block + self.overlap))
        self.fs_out = fs_out
        self.seen: dict[int, float] = {}
        self.samples_in = 0
        self.busy_s = 0.0

    def _emit(self, pkts: list[BlePacket]) -> list[BlePacket]:
        out = []
        for p in pkts:
            key = p.sample_index // (4 * self.rx.sps)
            if key in self.seen or key - 1 in self.seen or key + 1 in self.seen:
                continue
            self.seen[key] = 1
            p.extra["time_s"] = (p.sample_index - self.chan.delay_out) / self.fs_out  # remove FIR delay
            out.append(p)
        if len(self.seen) > 4096:
            for k in sorted(self.seen)[:2048]:
                del self.seen[k]
        return out

    def push(self, x: np.ndarray) -> list[BlePacket]:
        t0 = time.perf_counter()
        self.samples_in += len(x)
        y = self.chan.process(x)
        found = []
        # write in slices no larger than one block so a large source chunk can
        # never overrun the ring before it is decoded
        for i in range(0, len(y), self.block):
            self.ring.write(y[i:i + self.block])
            while len(self.ring) >= self.block + self.overlap:
                seg = self.ring.peek(self.block + self.overlap)
                found += self.rx.decode(seg, base_index=self.ring.read_index, start=0, stop=self.block)
                self.ring.consume(self.block)
        self.busy_s += time.perf_counter() - t0
        return self._emit(found)

    def flush(self) -> list[BlePacket]:
        seg = self.ring.peek()
        found = self.rx.decode(seg, base_index=self.ring.read_index) if len(seg) else []
        self.ring.consume(len(seg))
        return self._emit(found)

    def run(self, blocks: Iterable[np.ndarray]) -> Iterator[BlePacket]:
        for b in blocks:
            yield from self.push(b)
        yield from self.flush()
