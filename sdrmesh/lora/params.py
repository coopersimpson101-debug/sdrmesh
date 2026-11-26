"""LoRa PHY parameters and time-on-air.

Time on air follows the formula published by Semtech (SX1276/77/78/79
datasheet rev. 7, section 4.1.1.7; application note AN1200.13)::

    T_sym      = 2^SF / BW
    T_preamble = (n_preamble + 4.25) T_sym
    n_payload  = 8 + max(ceil((8 PL - 4 SF + 28 + 16 CRC - 20 IH) / (4 (SF - 2 DE))) (CR + 4), 0)
    T_packet   = T_preamble + n_payload T_sym

with CR = 1..4 for coding rates 4/5..4/8, IH = 1 for implicit header and
DE = 1 when low-data-rate optimisation is on (mandated when T_sym > 16 ms).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SYNC_PRIVATE = 0x12
SYNC_LORAWAN = 0x34


@dataclass(frozen=True)
class LoRaParams:
    sf: int = 7
    bw: float = 125e3
    cr: int = 1  # 1..4 -> 4/5..4/8
    has_crc: bool = True
    implicit_header: bool = False
    ldro: bool | None = None  # None = automatic (T_sym > 16 ms)
    preamble_len: int = 8
    sync_word: int = SYNC_PRIVATE
    os: int = 1  # oversampling factor: fs = os * bw

    def __post_init__(self):
        if not 7 <= self.sf <= 12:
            raise ValueError("SF must be 7..12")
        if not 1 <= self.cr <= 4:
            raise ValueError("CR must be 1..4 (4/5..4/8)")
        if self.preamble_len < 6:
            raise ValueError("preamble must be at least 6 symbols")
        if self.os < 1:
            raise ValueError("oversampling must be >= 1")

    @property
    def n(self) -> int:
        return 1 << self.sf

    @property
    def fs(self) -> float:
        return self.os * self.bw

    @property
    def t_sym(self) -> float:
        return self.n / self.bw

    @property
    def low_data_rate(self) -> bool:
        if self.ldro is not None:
            return self.ldro
        return self.t_sym > 16e-3

    @property
    def bitrate(self) -> float:
        """Nominal (coded) PHY bit rate SF * BW / 2^SF * 4 / (4 + CR)."""
        return self.sf * self.bw / self.n * 4 / (4 + self.cr)

    def payload_symbols(self, payload_len: int) -> int:
        de = 1 if self.low_data_rate else 0
        ih = 1 if self.implicit_header else 0
        crc = 1 if self.has_crc else 0
        num = 8 * payload_len - 4 * self.sf + 28 + 16 * crc - 20 * ih
        return 8 + max(math.ceil(num / (4 * (self.sf - 2 * de))) * (self.cr + 4), 0)

    def airtime(self, payload_len: int) -> float:
        """Time on air in seconds."""
        return (self.preamble_len + 4.25 + self.payload_symbols(payload_len)) * self.t_sym
