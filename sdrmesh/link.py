"""Long-range router-to-router link: meshfind frames over LoRa.

meshfind relays talk ESP-NOW (2.4 GHz, tens of metres). To bridge buildings
or sites, a router node can forward meshfind mesh frames (12-byte header +
108-byte encrypted report = 120 bytes) over a LoRa link. Because reports are
end-to-end encrypted by meshfind, the LoRa link needs no confidentiality of
its own; integrity against noise comes from the LoRa PHY CRC-16.

LR frame (little-endian), at most 255 bytes (the LoRa payload limit)::

    0  u8    magic 0x4C ('L')
    1  u8    version (4 bits, = 1) << 4 | type (1 = DATA, 2 = ACK)
    2  u16   source router id
    4  u16   destination router id (0xFFFF = broadcast)
    6  u8    sequence number
    7  u8    item count n
    8  ...   DATA: n x (u8 length | meshfind mesh frame)
             ACK:  n x u8 acknowledged sequence numbers

Two 120-byte report frames fit in one LoRa packet (8 + 2 x 121 = 250 bytes).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

from .lora.params import LoRaParams

MAGIC = 0x4C
VERSION = 1
T_DATA, T_ACK = 1, 2
HDR = struct.Struct("<BBHHBB")
BROADCAST = 0xFFFF
MAX_LORA_PAYLOAD = 255
MESHFIND_REPORT_FRAME = 120  # meshfind.mesh.HDR_LEN (12) + report.ENVELOPE_LEN (108)


class LinkError(ValueError):
    pass


@dataclass(frozen=True)
class LRFrame:
    ftype: int
    src: int
    dst: int
    seq: int
    items: list = field(default_factory=list)

    def encode(self) -> bytes:
        if self.ftype == T_DATA:
            body = b"".join(bytes([len(it)]) + bytes(it) for it in self.items)
        elif self.ftype == T_ACK:
            body = bytes(self.items)
        else:
            raise LinkError("unknown frame type")
        out = HDR.pack(MAGIC, (VERSION << 4) | self.ftype, self.src, self.dst, self.seq & 0xFF, len(self.items)) + body
        if len(out) > MAX_LORA_PAYLOAD:
            raise LinkError(f"frame is {len(out)} bytes, LoRa allows {MAX_LORA_PAYLOAD}")
        return out

    @classmethod
    def decode(cls, b: bytes) -> "LRFrame":
        if len(b) < HDR.size:
            raise LinkError("short frame")
        magic, vt, src, dst, seq, n = HDR.unpack(b[:HDR.size])
        if magic != MAGIC or vt >> 4 != VERSION:
            raise LinkError("bad magic/version")
        ftype = vt & 0xF
        body = b[HDR.size:]
        if ftype == T_ACK:
            if len(body) != n:
                raise LinkError("bad ack length")
            return cls(ftype, src, dst, seq, list(body))
        if ftype != T_DATA:
            raise LinkError("unknown frame type")
        items, i = [], 0
        for _ in range(n):
            if i >= len(body):
                raise LinkError("truncated item")
            ln = body[i]
            if i + 1 + ln > len(body):
                raise LinkError("truncated item")
            items.append(bytes(body[i + 1:i + 1 + ln]))
            i += 1 + ln
        if i != len(body):
            raise LinkError("trailing bytes")
        return cls(ftype, src, dst, seq, items)


def max_items(item_len: int = MESHFIND_REPORT_FRAME) -> int:
    return (MAX_LORA_PAYLOAD - HDR.size) // (1 + item_len)


def pack_reports(frames: list[bytes], src: int, dst: int, seq0: int = 0) -> list[LRFrame]:
    """Greedily bundle meshfind frames into as few LR frames as possible."""
    out, cur, size = [], [], HDR.size
    seq = seq0
    for f in frames:
        if size + 1 + len(f) > MAX_LORA_PAYLOAD and cur:
            out.append(LRFrame(T_DATA, src, dst, seq, cur))
            seq, cur, size = seq + 1, [], HDR.size
        cur.append(f)
        size += 1 + len(f)
    if cur:
        out.append(LRFrame(T_DATA, src, dst, seq, cur))
    return out


# ------------------------------------------------------------------ link budget
def noise_floor_dbm(bw: float, nf_db: float = 6.0) -> float:
    return -174.0 + 10 * math.log10(bw) + nf_db


def budget_row(p: LoRaParams, reports_per_frame: int, duty_cycle: float = 0.01,
               snr_req_db: float | None = None, tx_dbm: float = 14.0, nf_db: float = 6.0) -> dict:
    """Airtime / throughput for one configuration. ``snr_req_db`` (e.g. a
    simulated 10 %-PER threshold) gives a sensitivity estimate."""
    plen = HDR.size + reports_per_frame * (1 + MESHFIND_REPORT_FRAME)
    t = p.airtime(plen)
    row = {
        "sf": p.sf, "bw_khz": p.bw / 1e3, "cr": f"4/{4 + p.cr}", "ldro": p.low_data_rate,
        "payload_B": plen, "airtime_ms": t * 1e3,
        "reports_per_hour": duty_cycle * 3600 / t * reports_per_frame,
        "goodput_bps": reports_per_frame * 108 * 8 / t,
        "phy_bitrate_bps": p.bitrate,
    }
    if snr_req_db is not None:
        sens = noise_floor_dbm(p.bw, nf_db) + snr_req_db
        row.update(snr_req_db=snr_req_db, sensitivity_dbm=sens, max_path_loss_db=tx_dbm - sens)
    return row
