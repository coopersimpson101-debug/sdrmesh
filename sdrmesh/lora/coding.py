"""LoRa channel coding, bit-compatible (by construction) with the de facto
LoRa PHY as reverse-engineered in the literature.

Transmit chain (bytes -> chirp symbol values)::

    payload --whiten--> nibbles (low nibble first) --+
    header (5 nibbles, explicit mode) ---------------+--> Hamming(4, 4+CR)
    CRC-16 (4 nibbles, not whitened) ----------------+      |
                                                  diagonal interleaver (SF' x (4+CR) blocks)
                                                            |
                                         inverse Gray + 1  --> symbol value 0..2^SF-1

* The first block always uses CR 4/8 and SF' = SF - 2 ("reduced rate": the two
  LSBs of each symbol carry a parity bit and a zero, so +-1 bin errors do not
  corrupt the header). With low-data-rate optimisation every block is reduced.
* Header: ``len_hi | len_lo | CR << 1 | has_crc | chk4 | chk3..0`` with the
  5-bit checksum computed by the XOR equations in :func:`header_checksum`.
* CRC: CRC-16/CCITT (poly 0x1021, init 0) over all but the last two payload
  bytes, XORed with those two bytes (big-endian), sent as 4 nibbles LSB first.
* Whitening: the 8-bit LFSR ``x^8 + x^6 + x^5 + x^4 + 1`` started at 0xFF,
  XORed byte-wise over the payload only.

References
----------
* M. Knight, B. Seeber, "Decoding LoRa: Realizing a Modern LPWAN with SDR",
  GNU Radio Conf. 2016 (first public description of Gray mapping, diagonal
  interleaving and Hamming FEC).
* J. Tapparel, O. Afisiadis, P. Mayoraz, A. Balatsoukas-Stimming, A. Burg,
  "An Open-Source LoRa Physical Layer Prototype on GNU Radio", IEEE SPAWC 2020,
  and the gr-lora_sdr implementation (github.com/tapparelj/gr-lora_sdr), whose
  conventions (header checksum, CRC, whitening sequence, symbol +1 offset)
  this module follows. Code here is an independent implementation; it has
  been validated only against its own modulator and the published whitening
  table (see tests), not against over-the-air captures.
"""

from __future__ import annotations

import numpy as np


class LoRaDecodeError(ValueError):
    pass


# ---------------------------------------------------------------- whitening
def whitening_sequence(n: int) -> bytes:
    """LFSR x^8+x^6+x^5+x^4+1, seed 0xFF, new bit = b7^b5^b4^b3 shifted in at LSB."""
    out = bytearray()
    s = 0xFF
    for _ in range(n):
        out.append(s)
        fb = ((s >> 7) ^ (s >> 5) ^ (s >> 4) ^ (s >> 3)) & 1
        s = ((s << 1) | fb) & 0xFF
    return bytes(out)


def whiten(data: bytes) -> bytes:
    w = whitening_sequence(len(data))
    return bytes(a ^ b for a, b in zip(data, w))


# ---------------------------------------------------------------- header and CRC
def header_checksum(n0: int, n1: int, n2: int) -> int:
    """5-bit checksum over the first three header nibbles."""
    def b(v, k):
        return (v >> k) & 1
    c4 = b(n0, 3) ^ b(n0, 2) ^ b(n0, 1) ^ b(n0, 0)
    c3 = b(n0, 3) ^ b(n1, 3) ^ b(n1, 2) ^ b(n1, 1) ^ b(n2, 0)
    c2 = b(n0, 2) ^ b(n1, 3) ^ b(n1, 0) ^ b(n2, 3) ^ b(n2, 1)
    c1 = b(n0, 1) ^ b(n1, 2) ^ b(n1, 0) ^ b(n2, 2) ^ b(n2, 1) ^ b(n2, 0)
    c0 = b(n0, 0) ^ b(n1, 1) ^ b(n2, 3) ^ b(n2, 2) ^ b(n2, 1) ^ b(n2, 0)
    return (c4 << 4) | (c3 << 3) | (c2 << 2) | (c1 << 1) | c0


def header_nibbles(payload_len: int, cr: int, has_crc: bool) -> list[int]:
    if not 0 <= payload_len <= 255:
        raise ValueError("payload length must be 0..255")
    n0, n1, n2 = payload_len >> 4, payload_len & 0xF, (cr << 1) | int(has_crc)
    c = header_checksum(n0, n1, n2)
    return [n0, n1, n2, c >> 4, c & 0xF]


def parse_header(nib: list[int]) -> tuple[int, int, bool]:
    """-> (payload_len, cr, has_crc); raises LoRaDecodeError on a bad checksum."""
    n0, n1, n2, h, l = nib[:5]
    if ((h & 1) << 4 | l) != header_checksum(n0, n1, n2):
        raise LoRaDecodeError("header checksum mismatch")
    cr = n2 >> 1
    if not 1 <= cr <= 4:
        raise LoRaDecodeError("invalid coding rate in header")
    return (n0 << 4) | n1, cr, bool(n2 & 1)


def crc16_ccitt(data: bytes, crc: int = 0x0000) -> int:
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
            crc &= 0xFFFF
    return crc


def payload_crc(payload: bytes) -> int:
    crc = crc16_ccitt(payload[:-2]) if len(payload) >= 2 else crc16_ccitt(b"")
    if len(payload) >= 1:
        crc ^= payload[-1]
    if len(payload) >= 2:
        crc ^= payload[-2] << 8
    return crc


# ---------------------------------------------------------------- Hamming FEC
def hamming_encode(nibble: int, cr: int) -> int:
    """Codeword of ``4 + cr`` bits, written MSB first as d0 d1 d2 d3 p0 p1 p2 p3
    (d0 = data LSB), truncated to the code length; CR 4/5 uses one even-parity bit."""
    d0, d1, d2, d3 = (nibble >> 0) & 1, (nibble >> 1) & 1, (nibble >> 2) & 1, (nibble >> 3) & 1
    if cr == 1:
        p4 = d0 ^ d1 ^ d2 ^ d3
        return (d0 << 4) | (d1 << 3) | (d2 << 2) | (d3 << 1) | p4
    p0 = d0 ^ d1 ^ d2
    p1 = d1 ^ d2 ^ d3
    p2 = d0 ^ d1 ^ d3
    p3 = d0 ^ d2 ^ d3
    full = (d0 << 7) | (d1 << 6) | (d2 << 5) | (d3 << 4) | (p0 << 3) | (p1 << 2) | (p2 << 1) | p3
    return full >> (4 - cr)


_CODEBOOK = {cr: np.array([hamming_encode(v, cr) for v in range(16)]) for cr in (1, 2, 3, 4)}
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)])


def hamming_decode(cw: int, cr: int) -> tuple[int, bool]:
    """-> (nibble, corrected_or_detected_error).

    CR 4/7 and 4/8 (minimum distance 3 and 4): nearest-codeword decoding,
    which corrects any single bit error. CR 4/5 and 4/6 (distance 2) can only
    detect errors, so the systematic bits are returned and the flag set.
    """
    if cr <= 2:
        nib = ((cw >> (cr + 3)) & 1) | (((cw >> (cr + 2)) & 1) << 1) | (((cw >> (cr + 1)) & 1) << 2) | (((cw >> cr) & 1) << 3)
        return nib, bool(cw != hamming_encode(nib, cr))
    dist = _POPCOUNT[_CODEBOOK[cr] ^ cw]
    nib = int(np.argmin(dist))
    return nib, bool(dist[nib] != 0)


# ---------------------------------------------------------------- interleaver
def interleave_block(codewords: list[int], sf_app: int, cw_len: int, sf: int) -> list[int]:
    """Diagonal interleaver: ``sf_app`` codewords of ``cw_len`` bits -> ``cw_len``
    symbols of ``sf`` bits. Row ``i`` bit ``j`` (MSB first) is bit ``i`` of codeword
    ``(i - j - 1) mod sf_app``. Reduced-rate symbols get a parity bit and a 0
    in their two LSBs."""
    cw_bin = [[(c >> (cw_len - 1 - k)) & 1 for k in range(cw_len)] for c in codewords]
    out = []
    for i in range(cw_len):
        row = [cw_bin[(i - j - 1) % sf_app][i] for j in range(sf_app)]
        if sf_app == sf - 2:
            row += [sum(row) & 1, 0]
        v = 0
        for bit in row:
            v = (v << 1) | bit
        out.append(v)
    return out


def deinterleave_block(words: list[int], sf_app: int, cw_len: int) -> list[int]:
    """Inverse of :func:`interleave_block` on ``sf_app``-bit words."""
    cw_bin = [[0] * cw_len for _ in range(sf_app)]
    for i, w in enumerate(words):
        for j in range(sf_app):
            cw_bin[(i - j - 1) % sf_app][i] = (w >> (sf_app - 1 - j)) & 1
    out = []
    for bits in cw_bin:
        v = 0
        for b in bits:
            v = (v << 1) | b
        out.append(v)
    return out


# ---------------------------------------------------------------- Gray mapping
def gray(v: int) -> int:
    return v ^ (v >> 1)


def gray_inverse(g: int) -> int:
    v = g
    s = g >> 1
    while s:
        v ^= s
        s >>= 1
    return v


# ---------------------------------------------------------------- frame codec
def encode_frame(payload: bytes, sf: int, cr: int, has_crc: bool = True,
                 implicit_header: bool = False, ldro: bool = False) -> list[int]:
    """Payload bytes -> list of chirp symbol values (0..2^SF-1)."""
    nib: list[int] = []
    if not implicit_header:
        nib += header_nibbles(len(payload), cr, has_crc)
    for byte in whiten(payload):
        nib += [byte & 0xF, byte >> 4]
    if has_crc:
        c = payload_crc(payload)
        nib += [c & 0xF, (c >> 4) & 0xF, (c >> 8) & 0xF, (c >> 12) & 0xF]
    n_sym_bins = 1 << sf
    symbols: list[int] = []
    i = 0
    first = True
    while i < len(nib) or first:
        sf_app = sf - 2 if (first or ldro) else sf
        cr_app = 4 if first else cr
        block = nib[i:i + sf_app]
        cws = [hamming_encode(v, cr_app) for v in block] + [0] * (sf_app - len(block))
        for w in interleave_block(cws, sf_app, 4 + cr_app, sf):
            symbols.append((gray_inverse(w) + 1) % n_sym_bins)
        i += sf_app
        first = False
    return symbols


def symbols_to_words(bins: list[int], sf: int, reduced: bool) -> list[int]:
    """Demodulated FFT bins -> interleaver words (undo +1 offset and Gray)."""
    n = 1 << sf
    out = []
    for b in bins:
        s = (int(b) - 1) % n
        if reduced:
            s >>= 2
        out.append(gray(s))
    return out


def header_block_len() -> int:
    return 8  # 4 + CR(4/8)


def decode_header_block(bins: list[int], sf: int) -> tuple[list[int], int]:
    """First 8 symbols -> (sf-2 nibbles, number of corrected/detected errors)."""
    words = symbols_to_words(bins[:8], sf, reduced=True)
    cws = deinterleave_block(words, sf - 2, 8)
    out, errs = [], 0
    for c in cws:
        v, e = hamming_decode(c, 4)
        out.append(v)
        errs += e
    return out, errs


def decode_frame(bins: list[int], sf: int, cr: int | None = None, has_crc: bool | None = None,
                 implicit_len: int | None = None, ldro: bool = False) -> dict:
    """Chirp bins -> dict(payload, length, cr, has_crc, crc_ok, fec_errors).

    For implicit-header frames pass ``implicit_len``, ``cr`` and ``has_crc``.
    """
    if len(bins) < 8:
        raise LoRaDecodeError("need at least 8 symbols")
    nib, errs = decode_header_block(bins, sf)
    if implicit_len is None:
        length, cr_h, crc_h = parse_header(nib)
        cr, has_crc = cr_h, crc_h
    else:
        if cr is None or has_crc is None:
            raise ValueError("implicit header needs cr and has_crc")
        length = implicit_len
    need = (0 if implicit_len is not None else 5) + 2 * length + (4 if has_crc else 0)
    pos = 8
    sf_app = sf - 2 if ldro else sf
    cw_len = 4 + cr
    while len(nib) < need:
        blk = bins[pos:pos + cw_len]
        if len(blk) < cw_len:
            raise LoRaDecodeError("frame truncated")
        cws = deinterleave_block(symbols_to_words(blk, sf, ldro), sf_app, cw_len)
        for c in cws:
            v, e = hamming_decode(c, cr)
            nib.append(v)
            errs += e
        pos += cw_len
    off = 0 if implicit_len is not None else 5
    data = bytes((nib[off + 2 * k] | (nib[off + 2 * k + 1] << 4)) for k in range(length))
    payload = whiten(data)
    crc_ok = None
    if has_crc:
        c = nib[off + 2 * length: off + 2 * length + 4]
        rx_crc = c[0] | (c[1] << 4) | (c[2] << 8) | (c[3] << 12)
        crc_ok = rx_crc == payload_crc(payload)
    return {"payload": payload, "length": length, "cr": cr, "has_crc": has_crc,
            "crc_ok": crc_ok, "fec_errors": errs, "n_symbols": pos}
