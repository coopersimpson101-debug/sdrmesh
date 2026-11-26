"""BLE LE 1M receiver: channel filter -> FM discriminator -> sync-word
correlation -> CFO estimation/correction -> Gardner timing recovery ->
slicer -> de-whitening -> CRC-24 -> PDU.

Signal model after the channel filter (complex baseband, ``fs = sps * 1 MHz``)::

    r[n] = A exp(j (phi[n] + 2 pi f_o n / fs + theta)) + w[n]

1. **Front end**: a 600 kHz windowed-sinc channel filter, the discriminator
   ``d[n] = arg(r[n] r*[n-1]) * fs / (2 pi f_dev)`` (GFSK -> a +-1 PAM-like
   waveform plus a DC term ``f_o / f_dev``), then an integrate-and-dump
   (moving average over one symbol), which is the matched filter for the
   rectangular frequency pulse and gains several dB over sampling ``d`` raw.
2. **Sync**: normalised cross-correlation of ``d`` with the noiseless
   discriminator template of preamble + access address (40 bits). Removing
   the window mean makes the metric independent of the CFO, and dividing by
   the window energy makes it independent of the gain:
   ``rho[n] = sum_m d[n+m] t0[m] / (||t0|| sqrt(sum_m (d[n+m] - mu_n)^2))``,
   ``t0 = t - mean(t)``; ``t`` is the noiseless sync word passed through the
   same front end. Peaks above a threshold are candidate packets; a
   parabola through the peak gives sub-sample timing.
3. **CFO**: ``f_o ~ f_dev (mean(d over sync) - mean(t))`` (the known bits make
   this unbiased even though the preamble+AA is not DC-balanced), then the IQ
   is de-rotated by ``exp(-j 2 pi f_o n / fs)`` and re-discriminated.
4. **Timing**: Gardner loop (``timing.py``) started at the correlation peak.
5. **Bits**: sign slicer; access address checked with up to ``max_aa_errors``
   bit errors; the PDU header is de-whitened to read the length, then the PDU
   and CRC are read and the CRC-24 is checked.
"""

from __future__ import annotations

import math

import numpy as np

from ..dsp.filters import filt, freq_shift, lowpass
from .gfsk import modulate_bits
from .linklayer import (ADV_ACCESS_ADDRESS, ADV_CRC_INIT, bits_lsb_to_int, bits_to_bytes, crc24_bits,
                        sync_bits, whitening_sequence)
from .pdu import BlePacket
from .timing import GardnerTiming

F_DEV = 250e3
SYNC_LEN = 40


def discriminate(x: np.ndarray, fs: float, f_dev: float = F_DEV) -> np.ndarray:
    """FM discriminator, normalised to the nominal deviation. Same length as x."""
    x = np.asarray(x)
    d = np.empty(len(x))
    d[0] = 0.0
    d[1:] = np.angle(x[1:] * np.conj(x[:-1])) * fs / (2 * math.pi * f_dev)
    return d


_TPL_FFT_CACHE: dict = {}


def _xcorr_valid(d: np.ndarray, t0: np.ndarray) -> np.ndarray:
    """sum_m d[n+m] t0[m] for n = 0..len(d)-len(t0), via FFT (the template's
    spectrum is cached per FFT size)."""
    n = len(d) + len(t0) - 1
    nfft = 1 << (n - 1).bit_length()
    key = (hash(t0.tobytes()), len(t0), nfft)
    T = _TPL_FFT_CACHE.get(key)
    if T is None:
        T = np.conj(np.fft.rfft(t0, nfft))
        if len(_TPL_FFT_CACHE) > 64:
            _TPL_FFT_CACHE.clear()
        _TPL_FFT_CACHE[key] = T
    c = np.fft.irfft(np.fft.rfft(d, nfft) * T, nfft)
    return c[:len(d) - len(t0) + 1]


class BleReceiver:
    """Decode BLE LE 1M packets from complex baseband centred on one channel."""

    def __init__(self, fs: float = 4e6, channel: int = 37, access_address: int = ADV_ACCESS_ADDRESS,
                 crc_init: int = ADV_CRC_INIT, threshold: float = 0.55, max_aa_errors: int = 1,
                 lpf_cutoff: float = 600e3, max_pdu: int = 255, require_crc: bool = True,
                 post_len: int | None = None):
        sps = fs / 1e6
        if abs(sps - round(sps)) > 1e-9 or sps < 2:
            raise ValueError("fs must be an integer multiple (>= 2) of 1 MHz")
        self.fs = fs
        self.sps = int(round(sps))
        self.channel = channel
        self.aa = access_address
        self.crc_init = crc_init
        self.threshold = threshold
        self.max_aa_errors = max_aa_errors
        self.max_pdu = max_pdu
        self.require_crc = require_crc
        self.h = lowpass(8 * self.sps + 1, min(lpf_cutoff, 0.45 * fs), fs)
        n_post = post_len or self.sps
        self.post = np.ones(n_post) / n_post  # integrate-and-dump (default: one symbol)
        self.sync = sync_bits(access_address)
        # template = noiseless sync word through the *same* front end
        g = 8 + 2  # guard + ramp symbols added by modulate_bits
        ref = self._prepare(modulate_bits(self.sync, sps=self.sps, guard_symbols=8, ramp_symbols=2))
        tpl = ref[g * self.sps:(g + SYNC_LEN) * self.sps]
        self.tpl_mean = float(tpl.mean())
        self.t0 = tpl - self.tpl_mean
        self.t0_norm = float(np.linalg.norm(self.t0))
        self.wseq = whitening_sequence(channel, 8 * (2 + max_pdu + 3))
        self.t_offset = self._calibrate_delay()

    def _calibrate_delay(self) -> float:
        """Front-end delay (samples) from symbol start to the eye centre,
        measured on a noiseless random sequence: maximises the eye opening."""
        rng = np.random.default_rng(12345)
        bits = rng.integers(0, 2, 400).astype(np.uint8)
        g = 8 + 2
        d = self._prepare(modulate_bits(bits, sps=self.sps, guard_symbols=8, ramp_symbols=2))
        k = np.arange(20, 380)
        best, best_off = -1.0, 0.5 * self.sps
        for off in np.arange(0, self.sps, 0.05):
            pos = (g + k) * self.sps + off
            v = np.interp(pos, np.arange(len(d)), d) * (2.0 * bits[k] - 1.0)
            m = float(np.min(v))
            if m > best:
                best, best_off = m, float(off)
        return best_off

    # ------------------------------------------------------------ sync
    def correlate(self, d: np.ndarray) -> np.ndarray:
        L = len(self.t0)
        if len(d) < L:
            return np.zeros(0)
        num = _xcorr_valid(d, self.t0)
        c1 = np.concatenate([[0.0], np.cumsum(d)])
        c2 = np.concatenate([[0.0], np.cumsum(d * d)])
        s1 = c1[L:] - c1[:-L]
        s2 = c2[L:] - c2[:-L]
        var = np.maximum(s2 - s1 * s1 / L, 1e-12)
        return num / (self.t0_norm * np.sqrt(var))

    def find_sync(self, d: np.ndarray, start: int = 0) -> list[tuple[float, float]]:
        """Candidate (fractional_start_index, rho) pairs, strongest-first within
        each neighbourhood, in time order."""
        rho = self.correlate(d)
        if len(rho) == 0:
            return []
        above = np.flatnonzero(rho > self.threshold)
        out = []
        L = len(self.t0)
        i = 0
        while i < len(above):
            j = i
            # group indices that are within one sync-word length of each other
            while j + 1 < len(above) and above[j + 1] - above[i] < L // 2:
                j += 1
            seg = above[i:j + 1]
            k = int(seg[np.argmax(rho[seg])])
            frac = 0.0
            if 0 < k < len(rho) - 1:
                a, b, c = rho[k - 1], rho[k], rho[k + 1]
                den = a - 2 * b + c
                if den < 0:
                    frac = 0.5 * (a - c) / den
            if k >= start:
                out.append((k + frac, float(rho[k])))
            i = j + 1
        return out

    # ------------------------------------------------------------ demod
    def _prepare(self, x: np.ndarray) -> np.ndarray:
        """Channel filter -> FM discriminator -> integrate-and-dump."""
        return filt(discriminate(filt(np.asarray(x, dtype=complex), self.h), self.fs), self.post)

    def demod_from(self, x: np.ndarray, t_sync: float, d: np.ndarray | None = None,
                   cfo_norm: float | None = None):
        """Estimate CFO at a sync position, de-rotate, and return a Gardner
        timing object positioned on the first sync symbol plus the CFO (Hz)."""
        L = len(self.t0)
        i0 = int(round(t_sync))
        if cfo_norm is None:
            if d is None:
                d = self._prepare(x)
            cfo_norm = float(np.mean(d[i0:i0 + L])) - self.tpl_mean
        cfo_hz = cfo_norm * F_DEV
        # de-rotate a segment around the packet and re-discriminate
        lo = max(0, i0 - 4 * self.sps)
        hi = min(len(x), i0 + (SYNC_LEN + 8 * (2 + self.max_pdu + 3) + 8) * self.sps)
        seg = freq_shift(np.asarray(x[lo:hi], dtype=complex), -cfo_hz, self.fs, n0=lo)
        dseg = self._prepare(seg)
        t_first = (t_sync - lo) + self.t_offset
        return GardnerTiming(dseg, t_first, self.sps), cfo_hz

    def decode(self, x: np.ndarray, base_index: int = 0, start: int = 0,
               stop: int | None = None) -> list[BlePacket]:
        """Decode every packet whose sync word starts in ``[start, stop)``."""
        x = np.asarray(x, dtype=complex)
        d = self._prepare(x)
        pkts: list[BlePacket] = []
        next_free = start
        for t_sync, rho in self.find_sync(d, start):
            if stop is not None and t_sync >= stop:
                break
            if t_sync < next_free:
                continue
            pkt, end = self._decode_one(x, d, t_sync, base_index)
            if pkt is not None:
                pkt.extra["rho"] = rho
                pkts.append(pkt)
                next_free = end
        return pkts

    def _quick_aa_check(self, d, t_sync) -> bool:
        """Cheap rejection of false sync peaks: slice the 32 access-address bits
        at nominal timing (after removing the sync-window mean) before running
        CFO correction and the timing loop."""
        L = len(self.t0)
        i0 = int(round(t_sync))
        seg = d[i0:i0 + L]
        if len(seg) < L:
            return False
        pos = (t_sync + self.t_offset + self.sps * np.arange(8, SYNC_LEN)) - i0
        v = np.interp(pos, np.arange(L), seg) - (float(seg.mean()) - self.tpl_mean)
        errs = int(np.sum((v > 0).astype(np.uint8) != self.sync[8:]))
        return errs <= self.max_aa_errors + 4

    def _decode_one(self, x, d, t_sync, base_index):
        if not self._quick_aa_check(d, t_sync):
            return None, 0
        gt, cfo_hz = self.demod_from(x, t_sync, d)
        soft = gt.run(SYNC_LEN + 16)
        if len(soft) < SYNC_LEN + 16:
            return None, 0
        bits = (np.array(soft) > 0).astype(np.uint8)
        aa_err = int(np.sum(bits[8:SYNC_LEN] != self.sync[8:]))
        if aa_err > self.max_aa_errors:
            return None, 0
        hdr = bits[SYNC_LEN:SYNC_LEN + 16] ^ self.wseq[:16]
        length = bits_lsb_to_int(hdr[8:16])
        if length > self.max_pdu:
            return None, 0
        n_more = 8 * length + 24
        soft2 = gt.run(n_more)
        if len(soft2) < n_more:
            return None, 0
        body = np.concatenate([hdr, (np.array(soft2) > 0).astype(np.uint8) ^ self.wseq[16:16 + n_more]])
        pdu_bits = body[:16 + 8 * length]
        crc_rx = body[16 + 8 * length:]
        crc_ok = bool(np.array_equal(crc_rx, crc24_bits(pdu_bits, self.crc_init)))
        if self.require_crc and not crc_ok:
            return None, 0
        i0 = int(t_sync)
        n_samp = (SYNC_LEN + 16 + n_more) * self.sps
        p = float(np.mean(np.abs(x[i0:i0 + n_samp]) ** 2)) if n_samp else 0.0
        pkt = BlePacket(self.aa, bits_to_bytes(pdu_bits), crc_ok, base_index + i0, cfo_hz,
                        10 * math.log10(p + 1e-20), self.channel, aa_err)
        return pkt, i0 + n_samp

    # ------------------------------------------------------------ raw bits (BER tests)
    def raw_bits_after_sync(self, x: np.ndarray, n_bits: int, t_sync: float | None = None) -> np.ndarray | None:
        """Demodulate ``n_bits`` channel bits following the sync word (no
        de-whitening). Uses the strongest sync candidate if ``t_sync`` is None.
        Returns None if no sync word is found."""
        d = self._prepare(x)
        if t_sync is None:
            cands = self.find_sync(d)
            if not cands:
                return None
            t_sync = max(cands, key=lambda c: c[1])[0]
        gt, _ = self.demod_from(x, t_sync, d)
        soft = gt.run(SYNC_LEN + n_bits)
        bits = (np.array(soft[SYNC_LEN:]) > 0).astype(np.uint8)
        if len(bits) < n_bits:
            bits = np.concatenate([bits, np.zeros(n_bits - len(bits), dtype=np.uint8)])
        return bits
