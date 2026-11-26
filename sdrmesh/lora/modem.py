"""LoRa transmitter and receiver (frame sync, CFO/STO estimation, decoding).

Frame on air::

    | preamble_len x up-chirp(0) | sync1 | sync2 | 2.25 x down-chirp | header + payload symbols |

with ``sync1 = (sw >> 4) << 3`` and ``sync2 = (sw & 0xF) << 3`` (0x12 -> 8, 16).

Receiver (critically sampled after an anti-alias filter):

1. **Detection** - dechirp + FFT on a symbol-spaced grid; a preamble is
   ``>= 4`` consecutive windows whose peak bin agrees to +-2 (a half-bin CFO
   plus a half-chip timing offset splits the peak into two bins 2 apart).
2. **Fractional CFO** - the preamble is periodic with period N, so the
   dechirped peak advances in phase by ``2 pi c`` per window for a CFO of ``c``
   bins: ``c_frac = arg(sum X_i X_{i-1}^*) / 2 pi`` (Bernier et al. 2020).
3. **Integer CFO and STO** - an up-chirp window that starts ``tau`` samples
   late with CFO ``c`` peaks at ``k_up = c + tau``; a down-chirp dechirped with
   the up-chirp peaks at ``k_down = c - tau``. After re-aligning the grid by
   ``k_up`` (so ``tau = -c``) the down-chirps give ``k_down = 2c``.
4. **Sync word** - located on the re-aligned grid, which also fixes the
   symbol index of the header.
5. **Fractional STO** - after CFO removal the up-chirp peak sits ``-eps`` bins
   off its integer position; Jacobsen's three-bin estimator gives eps, and the
   decimation phase of an oversampled input is chosen to cancel it.

References: Tapparel et al. (SPAWC 2020); C. Bernier, F. Dehmas, N. Deparis,
"Low complexity LoRa frame synchronization for ultra-low power software-
defined radios", IEEE Trans. Commun. 68(5), 2020; E. Jacobsen, P. Kootsookos,
"Fast, accurate frequency estimators", IEEE Signal Process. Mag. 24(3), 2007.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..dsp.filters import filt, lowpass
from .chirp import downchirp, modulate_symbols, upchirp
from .coding import LoRaDecodeError, decode_frame, decode_header_block, encode_frame, parse_header
from .params import LoRaParams


def sync_symbols(sync_word: int) -> tuple[int, int]:
    return ((sync_word & 0xF0) >> 4) << 3, (sync_word & 0x0F) << 3


def modulate(payload: bytes, p: LoRaParams, pad_symbols: float = 2.0) -> np.ndarray:
    """Bytes -> complex baseband at ``p.fs`` (unit amplitude), with
    ``pad_symbols`` of silence before and after."""
    syms = encode_frame(payload, p.sf, p.cr, p.has_crc, p.implicit_header, p.low_data_rate)
    N, os = p.n, p.os
    s1, s2 = sync_symbols(p.sync_word)
    pre = modulate_symbols([0] * p.preamble_len + [s1, s2], p.sf, os)
    dc = downchirp(p.sf, os)
    down = np.concatenate([dc, dc, dc[:N * os // 4]])
    data = modulate_symbols(syms, p.sf, os)
    pad = np.zeros(int(pad_symbols * N * os), dtype=complex)
    return np.concatenate([pad, pre, down, data, pad])


def n_samples(payload_len: int, p: LoRaParams) -> int:
    """Samples of the frame itself (no padding): must equal airtime * fs."""
    return int(round(p.airtime(payload_len) * p.fs))


@dataclass
class LoRaFrame:
    payload: bytes
    crc_ok: bool | None
    length: int
    cr: int
    has_crc: bool
    start: int  # sample index (input rate) of the first preamble chirp estimate
    cfo_hz: float
    sto_frac: float
    snr_db: float
    fec_errors: int
    extra: dict = field(default_factory=dict)


def _signed(k: float, N: int) -> float:
    k = k % N
    return k - N if k > N / 2 else k


class LoRaReceiver:
    """Decode LoRa frames from complex baseband sampled at ``p.fs``."""

    def __init__(self, p: LoRaParams, min_run: int = 4):
        self.p = p
        self.N = p.n
        self.min_run = min_run
        self.dn = downchirp(p.sf)
        self.up = upchirp(p.sf)
        self.sw = sync_symbols(p.sync_word)
        if p.os > 1:
            self.h = lowpass(64 * p.os + 1, 0.6 * p.bw, p.fs)

    # -------------------------------------------------------------- helpers
    def _spectra(self, y: np.ndarray, start: int, count: int, ref: np.ndarray) -> np.ndarray:
        N = self.N
        start = max(0, start)
        count = max(0, min(count, (len(y) - start) // N))
        if count == 0:
            return np.zeros((0, N), dtype=complex)
        blk = y[start:start + count * N].reshape(count, N)
        return np.fft.fft(blk * ref[None, :], axis=1)

    def _find_runs(self, y: np.ndarray) -> list[int]:
        """Grid window indices where a preamble run of >= min_run windows ends."""
        X = self._spectra(y, 0, len(y) // self.N, self.dn)
        if len(X) == 0:
            return []
        mag = np.abs(X)
        k = np.argmax(mag, axis=1)
        pk = mag[np.arange(len(k)), k] ** 2
        noise = (np.sum(mag ** 2, axis=1) - pk) / (self.N - 1)
        strong = pk > 4 * noise  # loose energy gate; the run test does the real work
        runs = []
        run = 1
        for i in range(1, len(k)):
            d = (int(k[i]) - int(k[i - 1])) % self.N
            if strong[i] and strong[i - 1] and (d <= 2 or d >= self.N - 2):
                run += 1
            else:
                if run >= self.min_run:
                    runs.append(i - 1)
                run = 1
        if run >= self.min_run:
            runs.append(len(k) - 1)
        return [(r, int(k[r])) for r in runs]

    # -------------------------------------------------------------- main entry
    def decode(self, x: np.ndarray, implicit_len: int | None = None) -> list[LoRaFrame]:
        """Find and decode all frames in ``x``."""
        p, N = self.p, self.N
        x = np.asarray(x, dtype=complex)
        xf = filt(x, self.h) if p.os > 1 else x
        frames: list[LoRaFrame] = []
        offset = 0  # input-rate sample index from which to search
        while True:
            y = xf[offset::p.os]
            runs = self._find_runs(y)
            if not runs:
                break
            fr, end_chip = None, None
            for run_end, _k in runs:
                ph = self._best_phase(xf, offset, run_end) if p.os > 1 else 0
                try:
                    fr, end_chip = self._sync_and_decode(xf, xf[offset + ph::p.os], offset + ph, run_end,
                                                         implicit_len)
                except LoRaDecodeError:
                    fr = None
                if fr is not None:
                    break
            if fr is None:
                break
            frames.append(fr)
            offset = offset + ph + end_chip * p.os
            if offset >= len(xf) - 8 * N * p.os:
                break
        return frames

    def _best_phase(self, xf, offset, run_end) -> int:
        """Pick the decimation phase (of ``os``) whose preamble windows give the
        sharpest dechirped peak: this bounds the fractional timing error to
        half an input sample before CFO estimation (a half-chip error splits
        the peak and biases the phase-based fractional CFO estimate)."""
        p, N = self.p, self.N
        w0 = max(0, run_end - p.preamble_len + 2)
        best, best_ph = -1.0, 0
        for ph in range(p.os):
            X = self._spectra(xf[offset + ph::p.os], w0 * N, run_end - w0, self.dn)
            if len(X) == 0:
                continue
            m = np.abs(X) ** 2
            score = float(np.mean(m.max(axis=1) / m.sum(axis=1)))
            if score > best:
                best, best_ph = score, ph
        return best_ph

    def _sync_and_decode(self, xf, y, offset, run_end, implicit_len):
        p, N = self.p, self.N
        n = np.arange(len(y))
        # windows inside the run (skip the last one, it may straddle the sync word)
        w_last = run_end - 1
        w_first = max(0, run_end - p.preamble_len + 2)
        X = self._spectra(y, w_first * N, w_last - w_first + 1, self.dn)
        if len(X) < 2:
            return None, None
        kk = np.argmax(np.abs(X), axis=1)
        k0 = int(np.bincount(kk, minlength=N).argmax())
        nb = [(k0 - 1) % N, k0, (k0 + 1) % N]  # peak and neighbours: robust to a split peak
        pk = X[:, nb]
        c_frac = float(np.angle(np.sum(pk[1:] * np.conj(pk[:-1]))) / (2 * math.pi))
        rot = np.exp(-2j * math.pi * c_frac * n / N)
        yc = y * rot
        # integer peak of the preamble after fractional CFO removal
        Xc = self._spectra(yc, w_first * N, w_last - w_first + 1, self.dn)
        k_up = int(np.bincount(np.argmax(np.abs(Xc), axis=1), minlength=N).argmax())
        # re-align the grid: window start s0 + m N has the preamble at bin 0
        s0 = (w_first * N - k_up) % N
        n_win = (len(yc) - s0) // N
        bins = np.argmax(np.abs(self._spectra(yc, s0, n_win, self.dn)), axis=1)
        m_lo = max(0, (w_first * N - k_up - s0) // N)
        m_sync = None
        best = -1
        for m in range(m_lo, min(m_lo + p.preamble_len + 4, n_win - 3)):
            sc = sum(min((int(bins[m + j]) - v) % N, (v - int(bins[m + j])) % N) <= 2
                     for j, v in enumerate(self.sw))
            b_prev = int(bins[m - 1 if m > 0 else m])
            if (b_prev <= 2 or b_prev >= N - 2) and sc > best:
                best, m_sync = sc, m
                if sc == 2:
                    break
        if m_sync is None or best < 1:
            return None, None
        # down-chirps: windows m_sync + 2, m_sync + 3, dechirped with the up-chirp
        D = self._spectra(yc, s0 + (m_sync + 2) * N, 2, self.up)
        if len(D) < 2:
            return None, None
        kd = np.argmax(np.abs(D[0]) ** 2 + np.abs(D[1]) ** 2)
        c_int = int(round(_signed(kd, N) / 2))
        # symbol boundary: window start was tau = -c_int samples off
        sym0 = s0 + c_int  # chip index of a symbol boundary on the corrected grid
        cfo_bins = c_int + c_frac
        # fractional STO from the preamble peaks (Jacobsen), after full CFO removal
        yc2 = y * np.exp(-2j * math.pi * cfo_bins * n / N)
        pre_start = sym0 + (m_sync - (p.preamble_len - 1)) * N
        P = self._spectra(yc2, max(pre_start, sym0 + m_lo * N), min(p.preamble_len - 2, 4), self.dn)
        eps = 0.0
        if len(P):
            a = P[:, -1]
            b = P[:, 0]
            c = P[:, 1]
            den = 2 * b - a - c
            with np.errstate(divide="ignore", invalid="ignore"):
                dlt = np.real((a - c) / den)
            dlt = dlt[np.isfinite(dlt)]
            if len(dlt):
                eps = float(np.clip(np.median(dlt), -0.5, 0.5))
        # header starts 4.25 symbols after the first sync word
        hdr_chip = sym0 + m_sync * N + int(4.25 * N)
        if p.os > 1:
            shift = int(round(eps * p.os))  # positive delta -> sample earlier
            start = offset + hdr_chip * p.os - shift
            if start < 0:
                start += p.os
            idx_n = np.arange(start, len(xf))
            cf = xf[start:] * np.exp(-2j * math.pi * cfo_bins * (idx_n - offset) / (N * p.os))
            z = cf[::p.os]
        else:
            z = yc2[hdr_chip:]
        # demodulate header block, then as many symbols as the header implies
        hb = np.argmax(np.abs(self._spectra(z, 0, 8, self.dn)), axis=1)
        if len(hb) < 8:
            return None, None
        if implicit_len is None:
            nib, _ = decode_header_block(list(hb), p.sf)
            length, cr, has_crc = parse_header(nib)
        else:
            length, cr, has_crc = implicit_len, p.cr, p.has_crc
        pp = LoRaParams(p.sf, p.bw, cr, has_crc, implicit_len is not None, p.low_data_rate)
        n_sym = pp.payload_symbols(length)
        S = self._spectra(z, 0, n_sym, self.dn)
        if len(S) < n_sym:
            raise LoRaDecodeError("frame truncated")
        mag2 = np.abs(S) ** 2
        allb = np.argmax(mag2, axis=1)
        res = decode_frame(list(allb), p.sf, cr=cr, has_crc=has_crc,
                           implicit_len=implicit_len, ldro=p.low_data_rate)
        sig = mag2[np.arange(n_sym), allb]
        noise = (mag2.sum(axis=1) - sig) / (N - 1)
        snr = 10 * math.log10(max(np.mean(sig - noise), 1e-20) / max(np.mean(noise), 1e-20) / N)
        start_pre = offset + (sym0 + (m_sync - p.preamble_len) * N) * p.os
        frame = LoRaFrame(res["payload"], res["crc_ok"], length, cr, has_crc, int(start_pre),
                          cfo_bins * p.bw / N, eps, snr, res["fec_errors"],
                          extra={"cfo_bins": cfo_bins, "c_int": c_int})
        end_chip = hdr_chip + n_sym * N
        return frame, end_chip
