"""Glue between the BLE PHY and meshfind (github.com/coopersimpson101-debug/meshfind).

meshfind tags send a legacy ADV_NONCONN_IND whose static-random address is
``P_i[0..5] | 0xC0`` and whose 31-byte AdvData is one Manufacturer Specific
Data structure carrying the rest of the rotating X25519 key ``P_i``. This
module builds such adverts with meshfind's own reference code and turns
received :class:`~sdrmesh.ble.pdu.BlePacket` objects back into
``meshfind.adv.TagAdvert``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .ble.gfsk import modulate_pdu
from .ble.pdu import ADV_NONCONN_IND, BlePacket, adv_nonconn
from .channel import awgn, sigma2_for_ebn0
from .dsp.filters import freq_shift
from .dsp.interp import fractional_delay_fft

try:  # meshfind is an optional dependency (pip install "sdrmesh[meshfind]")
    from meshfind import adv as mf_adv
    from meshfind import ratchet as mf_ratchet
    HAVE_MESHFIND = True
except ImportError:  # pragma: no cover
    mf_adv = mf_ratchet = None
    HAVE_MESHFIND = False


def _need():
    if not HAVE_MESHFIND:
        raise RuntimeError("meshfind is not installed: pip install 'sdrmesh[meshfind]'")


def tag_advert(master: bytes, t: float, epoch0: int = 0, battery: int = 0, separated: bool = False):
    """The (address, payload, public_key) a meshfind tag provisioned with
    ``master`` would advertise at time ``t``, computed by meshfind itself."""
    _need()
    st = mf_ratchet.TagState.provision(master, epoch0)
    pub = st.current_public(t, separated)
    addr, payload = mf_adv.encode(mf_adv.TagAdvert(pub, battery, separated))
    return addr, payload, pub


def decode_meshfind(pkt: BlePacket):
    """-> meshfind TagAdvert or None if the packet is not a meshfind advert."""
    _need()
    a = pkt.adv
    if a is None or a.pdu_type != ADV_NONCONN_IND or not a.tx_add or a.adv_address is None:
        return None
    try:
        return mf_adv.decode(a.adv_address, a.adv_data)
    except mf_adv.AdvError:
        return None


def key_hash_hex(pub: bytes) -> str:
    _need()
    return mf_ratchet.key_hash(pub).hex()


@dataclass
class TruthAdvert:
    sample_index: int
    public_key: bytes
    address: bytes
    cfo_hz: float


def synth_capture(n_adverts: int, fs: float = 4e6, ebn0_db: float | None = 20.0, channel: int = 37,
                  cfo_max_hz: float = 100e3, gap_us: tuple[float, float] = (300, 1500),
                  seed: int = 1, n_tags: int = 3, extra_bad: bool = True):
    """Synthesise a capture of ``n_adverts`` meshfind adverts from ``n_tags``
    tags (each with its own CFO), random gaps, AWGN at ``ebn0_db`` (None = no
    noise, e.g. for transmitting with ``hackrf_transfer -t``), plus one
    non-meshfind advert.

    Returns ``(iq, truth)``.
    """
    _need()
    rng = np.random.default_rng(seed)
    sps = int(round(fs / 1e6))
    masters = [bytes(rng.integers(0, 256, 32, dtype=np.uint8)) for _ in range(n_tags)]
    cfos = rng.uniform(-cfo_max_hz, cfo_max_hz, n_tags)
    parts, truth = [], []
    pos = int(200e-6 * fs)
    parts.append(np.zeros(pos, dtype=complex))
    for i in range(n_adverts):
        tag = i % n_tags
        addr, payload, pub = tag_advert(masters[tag], t=1000.0 * i, battery=int(rng.integers(0, 4)))
        x = modulate_pdu(adv_nonconn(addr, payload), channel, sps=sps, guard_symbols=0)
        x = fractional_delay_fft(x, float(rng.uniform(0, 1)))
        x = freq_shift(x, cfos[tag], fs, n0=pos) * np.exp(1j * rng.uniform(0, 2 * math.pi))
        # truth index = first preamble sample (after the 2-symbol carrier ramp)
        truth.append(TruthAdvert(pos + 2 * sps, pub, addr, float(cfos[tag])))
        parts.append(x)
        pos += len(x)
        gap = np.zeros(int(rng.uniform(*gap_us) * 1e-6 * fs), dtype=complex)
        parts.append(gap)
        pos += len(gap)
        if extra_bad and i == n_adverts // 2:
            # a non-meshfind advert (public address, a name AD structure) to show filtering
            other = adv_nonconn(bytes.fromhex("001122334455"), bytes([0x08, 0x09]) + b"sensor1",
                                random_address=False)
            y = modulate_pdu(other, channel, sps=sps, guard_symbols=0)
            parts.append(y)
            pos += len(y)
            parts.append(np.zeros(int(400e-6 * fs), dtype=complex))
            pos += int(400e-6 * fs)
    sig = np.concatenate(parts)
    if ebn0_db is None:
        return sig, truth
    return awgn(sig, sigma2_for_ebn0(ebn0_db, sps), rng), truth
