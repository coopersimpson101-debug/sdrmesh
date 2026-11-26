"""Monte Carlo BER / SER / PER measurements through the full receivers.

Every point is a real simulation: random bits, the modulator, the channel
model (``channel.py``) and the receiver, counting errors. Nothing here is
fitted or interpolated.
"""

from __future__ import annotations

import math

import numpy as np

from . import channel as ch
from .ble.gfsk import modulate_bits
from .ble.linklayer import sync_bits
from .ble.receiver import BleReceiver
from .lora.chirp import downchirp, modulate_symbols
from .lora.modem import LoRaReceiver, modulate
from .lora.params import LoRaParams

BLE_IMPAIRMENTS = {
    "awgn": {},
    "cfo+sto": {"cfo_hz": 120e3, "delay": 0.37, "ppm": 40.0},
    "multipath": {"cfo_hz": 50e3, "delay": 0.37, "taps": ([0.0, 1.0, 2.5], [0.85, 0.45j, -0.25])},
}


def ble_ber_point(ebn0_db: float, n_frames: int, frame_bits: int = 1000, fs: float = 4e6,
                  impairment: str = "awgn", seed: int = 0) -> dict:
    """Raw channel BER of the GFSK demodulator *including* sync: each frame is
    preamble + AA + ``frame_bits`` random bits; a missed sync counts every bit
    of that frame as an error with probability 1/2 (i.e. adds frame_bits/2)."""
    rng = np.random.default_rng(seed)
    sps = int(fs / 1e6)
    rx = BleReceiver(fs)
    imp = dict(BLE_IMPAIRMENTS[impairment])
    errors = 0.0
    misses = 0
    for _ in range(n_frames):
        data = rng.integers(0, 2, frame_bits).astype(np.uint8)
        x = modulate_bits(np.concatenate([sync_bits(), data]), sps=sps, guard_symbols=0)
        taps = imp.get("taps")
        if taps is not None:
            g = np.array(taps[1], dtype=complex)
            taps = (taps[0], list(g / np.sqrt(np.sum(np.abs(g) ** 2))))
        y = ch.apply(x, fs, rng, cfo_hz=imp.get("cfo_hz", 0.0), delay=imp.get("delay", 0.0),
                     ppm=imp.get("ppm", 0.0), taps=taps, lead=50 * sps, tail=20 * sps,
                     sigma2=ch.sigma2_for_ebn0(ebn0_db, sps))
        bits = rx.raw_bits_after_sync(y, frame_bits)
        if bits is None:
            misses += 1
            errors += frame_bits / 2
        else:
            errors += int(np.sum(bits != data))
    n = n_frames * frame_bits
    return {"ebn0_db": ebn0_db, "bits": n, "errors": errors, "ber": errors / n, "sync_miss": misses,
            "impairment": impairment}


def lora_ser_point(sf: int, snr_db: float, n_sym: int, seed: int = 0, batch: int = 512) -> dict:
    """Uncoded symbol error rate with ideal synchronisation (dechirp + FFT
    argmax), critically sampled, AWGN only."""
    rng = np.random.default_rng(seed)
    N = 1 << sf
    dn = downchirp(sf)
    sigma2 = ch.sigma2_for_snr(snr_db, 1.0, 1.0)
    err_sym = 0
    err_bits = 0
    done = 0
    while done < n_sym:
        m = min(batch, n_sym - done)
        s = rng.integers(0, N, m)
        x = modulate_symbols(s, sf).reshape(m, N)
        x = x + math.sqrt(sigma2 / 2) * (rng.standard_normal((m, N)) + 1j * rng.standard_normal((m, N)))
        k = np.argmax(np.abs(np.fft.fft(x * dn[None, :], axis=1)), axis=1)
        wrong = k != s
        err_sym += int(np.sum(wrong))
        # Gray-coded bits as in the real chain: bits of gray(k-1) vs gray(s-1)
        gk = ((k - 1) % N) ^ (((k - 1) % N) >> 1)
        gs = ((s - 1) % N) ^ (((s - 1) % N) >> 1)
        err_bits += int(sum(bin(int(v)).count("1") for v in (gk ^ gs)[wrong]))
        done += m
    return {"sf": sf, "snr_db": snr_db, "symbols": n_sym, "symbol_errors": err_sym,
            "ser": err_sym / n_sym, "ber": err_bits / (n_sym * sf)}


def lora_per_point(sf: int, cr: int, snr_db: float, n_pkts: int, payload_len: int = 20,
                   bw: float = 125e3, os: int = 1, cfo_bins_max: float = 0.0, seed: int = 0) -> dict:
    """Packet error rate through the full receiver (detection, CFO/STO sync,
    header, FEC, CRC). A packet counts as received only if the payload matches
    and the CRC passes."""
    rng = np.random.default_rng(seed)
    p = LoRaParams(sf=sf, bw=bw, cr=cr, os=os)
    rx = LoRaReceiver(p)
    ok = 0
    sigma2 = ch.sigma2_for_snr(snr_db, p.fs, p.bw)
    for _ in range(n_pkts):
        pl = bytes(rng.integers(0, 256, payload_len, dtype=np.uint8))
        x = modulate(pl, p, pad_symbols=0)
        cfo = float(rng.uniform(-cfo_bins_max, cfo_bins_max)) * p.bw / p.n
        lead = int(rng.integers(p.n * os, 3 * p.n * os))
        y = ch.apply(x, p.fs, rng, cfo_hz=cfo, delay=float(rng.uniform(0, 1)) * os if os > 1 else 0.0,
                     lead=lead, tail=4 * p.n * os, sigma2=sigma2)
        fr = rx.decode(y)
        if fr and fr[0].payload == pl and fr[0].crc_ok:
            ok += 1
    return {"sf": sf, "cr": cr, "snr_db": snr_db, "packets": n_pkts, "per": 1 - ok / n_pkts,
            "payload_len": payload_len, "bw": bw, "os": os, "cfo_bins_max": cfo_bins_max}
