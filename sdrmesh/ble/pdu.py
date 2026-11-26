"""Advertising-channel PDUs (Core Spec v5.4, Vol 6, Part B, 2.3).

Header (16 bits, LSB first)::

    bits 0..3  PDU type     bit 5  ChSel   bit 6  TxAdd   bit 7  RxAdd
    bit  4     RFU          bits 8..15  Length (payload octets)

AdvA and other device addresses are carried least-significant octet first,
so the display form ``C1:22:33:44:55:66`` is the reverse of the PDU bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ADV_IND, ADV_DIRECT_IND, ADV_NONCONN_IND, SCAN_REQ, SCAN_RSP, CONNECT_IND, ADV_SCAN_IND = range(7)
PDU_NAMES = {
    ADV_IND: "ADV_IND", ADV_DIRECT_IND: "ADV_DIRECT_IND", ADV_NONCONN_IND: "ADV_NONCONN_IND",
    SCAN_REQ: "SCAN_REQ", SCAN_RSP: "SCAN_RSP", CONNECT_IND: "CONNECT_IND", ADV_SCAN_IND: "ADV_SCAN_IND",
    7: "ADV_EXT_IND",
}
# PDU types whose payload begins with the advertiser's address
_HAS_ADVA = {ADV_IND, ADV_DIRECT_IND, ADV_NONCONN_IND, SCAN_RSP, ADV_SCAN_IND}


class PduError(ValueError):
    pass


@dataclass(frozen=True)
class AdvPdu:
    pdu_type: int
    payload: bytes
    tx_add: bool = True
    rx_add: bool = False
    ch_sel: bool = False

    @property
    def name(self) -> str:
        return PDU_NAMES.get(self.pdu_type, f"type{self.pdu_type}")

    @property
    def adv_address(self) -> bytes | None:
        """AdvA in display (MSB-first) order, if this PDU type carries one."""
        if self.pdu_type in _HAS_ADVA and len(self.payload) >= 6:
            return self.payload[5::-1]
        return None

    @property
    def adv_data(self) -> bytes:
        if self.pdu_type in (ADV_IND, ADV_NONCONN_IND, SCAN_RSP, ADV_SCAN_IND):
            return self.payload[6:]
        return b""

    def encode(self) -> bytes:
        if len(self.payload) > 255:
            raise PduError("payload too long")
        h0 = (self.pdu_type & 0xF) | (int(self.ch_sel) << 5) | (int(self.tx_add) << 6) | (int(self.rx_add) << 7)
        return bytes([h0, len(self.payload)]) + self.payload

    @classmethod
    def decode(cls, pdu: bytes) -> "AdvPdu":
        if len(pdu) < 2:
            raise PduError("PDU shorter than its header")
        h0, length = pdu[0], pdu[1]
        if len(pdu) != 2 + length:
            raise PduError("length field does not match PDU size")
        return cls(h0 & 0xF, bytes(pdu[2:]), bool(h0 & 0x40), bool(h0 & 0x80), bool(h0 & 0x20))


def adv_nonconn(address_msb_first: bytes, adv_data: bytes, random_address: bool = True) -> AdvPdu:
    """Build an ADV_NONCONN_IND (what meshfind tags send)."""
    if len(address_msb_first) != 6:
        raise PduError("address must be 6 bytes")
    if len(adv_data) > 31:
        raise PduError("legacy AdvData is at most 31 octets")
    return AdvPdu(ADV_NONCONN_IND, bytes(address_msb_first[::-1]) + bytes(adv_data), tx_add=random_address)


@dataclass
class AdStructure:
    ad_type: int
    data: bytes


def parse_ad(adv_data: bytes) -> list[AdStructure]:
    """Split AdvData into AD structures (length | type | data). Stops at a zero
    length octet (significant-part terminator); raises on truncation."""
    out: list[AdStructure] = []
    i = 0
    while i < len(adv_data):
        n = adv_data[i]
        if n == 0:
            break
        if i + 1 + n > len(adv_data):
            raise PduError("truncated AD structure")
        out.append(AdStructure(adv_data[i + 1], bytes(adv_data[i + 2:i + 1 + n])))
        i += 1 + n
    return out


@dataclass
class BlePacket:
    """A decoded air packet plus receiver measurements."""

    access_address: int
    pdu: bytes
    crc_ok: bool
    sample_index: int
    cfo_hz: float
    rssi_db: float
    channel: int
    aa_errors: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def adv(self) -> AdvPdu | None:
        try:
            return AdvPdu.decode(self.pdu)
        except PduError:
            return None
