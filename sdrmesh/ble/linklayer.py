"""BLE link-layer bit processing: bit order, CRC-24, data whitening, air packets.

Written from the Bluetooth Core Specification v5.4, Vol 6 (Low Energy
Controller), Part B:

* 2.1   packet format (LE 1M uncoded): preamble (1 octet), access address
  (4 octets), PDU (2..257 octets), CRC (3 octets). All multi-bit fields are
  sent least-significant bit first, *except* the CRC (most significant first).
* 2.1.1 preamble: 0xAA if the access address LSB is 0, else 0x55, so the last
  preamble bit differs from the first access-address bit.
* 3.1.1 CRC: 24-bit LFSR, polynomial x^24 + x^10 + x^9 + x^6 + x^4 + x^3 + x + 1,
  preset 0x555555 on the primary advertising channels (position 0 = LSB of the
  preset); computed over the PDU; sent from position 23 down to position 0.
* 3.2   whitening: 7-bit LFSR x^7 + x^4 + 1, position 0 set to 1 and positions
  1..6 loaded with the channel index (MSB in position 1); applied to PDU and CRC.

Bits are handled as ``numpy.uint8`` arrays of 0/1 in on-air order.
"""

from __future__ import annotations

import numpy as np

ADV_ACCESS_ADDRESS = 0x8E89BED6
ADV_CRC_INIT = 0x555555
CRC_POLY = 0x00065B  # taps at positions 0,1,3,4,6,9,10 (x^24 implicit)

# Primary advertising channels: channel index -> centre frequency (Hz)
ADV_CHANNELS = {37: 2402e6, 38: 2426e6, 39: 2480e6}


def channel_freq(index: int) -> float:
    """RF centre frequency of BLE channel index 0..39 (Vol 6 Part B 1.4.1)."""
    if index in ADV_CHANNELS:
        return ADV_CHANNELS[index]
    if 0 <= index <= 10:
        return 2404e6 + 2e6 * index
    if 11 <= index <= 36:
        return 2428e6 + 2e6 * (index - 11)
    raise ValueError("channel index must be 0..39")


def bytes_to_bits(data: bytes) -> np.ndarray:
    """LSB-first serialisation of each octet."""
    a = np.frombuffer(bytes(data), dtype=np.uint8)
    return ((a[:, None] >> np.arange(8)[None, :]) & 1).astype(np.uint8).ravel()


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Inverse of :func:`bytes_to_bits` (length must be a multiple of 8)."""
    b = np.asarray(bits, dtype=np.uint8).reshape(-1, 8)
    return bytes((b << np.arange(8)[None, :]).sum(axis=1).astype(np.uint8))


def int_to_bits_lsb(value: int, n: int) -> np.ndarray:
    return np.array([(value >> i) & 1 for i in range(n)], dtype=np.uint8)


def bits_lsb_to_int(bits) -> int:
    return int(sum(int(b) << i for i, b in enumerate(bits)))


# ---------------------------------------------------------------- CRC-24
def crc24_register(pdu_bits: np.ndarray, init: int = ADV_CRC_INIT) -> int:
    """Run the spec LFSR over on-air PDU bits; returns the 24-bit register with
    bit k = LFSR position k."""
    r = init & 0xFFFFFF
    for b in pdu_bits:
        fb = ((r >> 23) & 1) ^ int(b)
        r = (r << 1) & 0xFFFFFF
        if fb:
            r ^= CRC_POLY
    return r


def crc24_bits(pdu_bits: np.ndarray, init: int = ADV_CRC_INIT) -> np.ndarray:
    """The 24 CRC bits in transmission order (position 23 first)."""
    r = crc24_register(pdu_bits, init)
    return np.array([(r >> (23 - i)) & 1 for i in range(24)], dtype=np.uint8)


def _reverse24(v: int) -> int:
    return int(f"{v & 0xFFFFFF:024b}"[::-1], 2)


def _crc24_reflected_table() -> np.ndarray:
    """Byte-wise table for the bit-reversed form of the same LFSR (an
    independent formulation used to cross-check :func:`crc24_register`)."""
    poly_r = _reverse24(CRC_POLY)  # bit-reversed taps; feedback now enters at bit 23
    table = np.zeros(256, dtype=np.uint32)
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ poly_r if c & 1 else c >> 1
        table[i] = c
    return table


_TABLE = _crc24_reflected_table()


def crc24_bytes_fast(pdu: bytes, init: int = ADV_CRC_INIT) -> int:
    """Table-driven CRC on whole octets (LSB-first bit order). Returns the
    register in the same orientation as :func:`crc24_register`."""
    c = _reverse24(init)
    for byte in pdu:
        c = (c >> 8) ^ int(_TABLE[(c ^ byte) & 0xFF])
    return _reverse24(c)


# ---------------------------------------------------------------- whitening
def whitening_sequence(channel: int, n: int) -> np.ndarray:
    """First ``n`` bits of the whitening sequence for ``channel`` (0..39)."""
    if not 0 <= channel <= 39:
        raise ValueError("channel index must be 0..39")
    # state[k] = LFSR position k
    state = [1] + [(channel >> (5 - k)) & 1 for k in range(6)]
    out = np.empty(n, dtype=np.uint8)
    for i in range(n):
        o = state[6]
        out[i] = o
        # shift: pos k <- pos k-1, pos0 <- pos6, pos4 <- pos3 ^ pos6
        new = [o] + state[:6]
        new[4] ^= o
        state = new
    return out


def whiten(bits: np.ndarray, channel: int) -> np.ndarray:
    """Whitening is its own inverse."""
    bits = np.asarray(bits, dtype=np.uint8)
    return bits ^ whitening_sequence(channel, len(bits))


# ---------------------------------------------------------------- air packet
def preamble_bits(access_address: int) -> np.ndarray:
    return int_to_bits_lsb(0xAA if (access_address & 1) == 0 else 0x55, 8)


def build_air_bits(pdu: bytes, channel: int = 37, access_address: int = ADV_ACCESS_ADDRESS,
                   crc_init: int = ADV_CRC_INIT) -> np.ndarray:
    """Complete on-air bit stream: preamble | AA | whiten(PDU | CRC)."""
    pdu_bits = bytes_to_bits(pdu)
    crc = crc24_bits(pdu_bits, crc_init)
    body = whiten(np.concatenate([pdu_bits, crc]), channel)
    return np.concatenate([preamble_bits(access_address), int_to_bits_lsb(access_address, 32), body])


def sync_bits(access_address: int = ADV_ACCESS_ADDRESS) -> np.ndarray:
    """Preamble + access address: the 40-bit synchronisation word."""
    return np.concatenate([preamble_bits(access_address), int_to_bits_lsb(access_address, 32)])
