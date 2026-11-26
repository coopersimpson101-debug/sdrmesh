import math
import os

import numpy as np
import pytest

from sdrmesh import channel as ch
from sdrmesh.ble import linklayer as ll
from sdrmesh.ble.gfsk import frequency_waveform, modulate_bits, modulate_pdu
from sdrmesh.ble.pdu import ADV_IND, ADV_NONCONN_IND, AdvPdu, PduError, adv_nonconn, parse_ad
from sdrmesh.ble.receiver import BleReceiver, discriminate
from sdrmesh.ble.timing import GardnerTiming
from sdrmesh.dsp import resample_ppm


# ---------------------------------------------------------------- link layer
def test_bit_order_roundtrip():
    data = bytes(range(0, 256, 7))
    bits = ll.bytes_to_bits(data)
    assert list(bits[:8]) == [0, 0, 0, 0, 0, 0, 0, 0] and list(ll.bytes_to_bits(b"\x01")) == [1, 0, 0, 0, 0, 0, 0, 0]
    assert ll.bits_to_bytes(bits) == data


def test_preamble_rule():
    # AA LSB 0 -> 0xAA (bits 0,1,0,1,...): last preamble bit differs from first AA bit
    pre = ll.preamble_bits(ll.ADV_ACCESS_ADDRESS)
    assert list(pre) == [0, 1, 0, 1, 0, 1, 0, 1]
    assert pre[-1] != ll.int_to_bits_lsb(ll.ADV_ACCESS_ADDRESS, 32)[0]
    assert list(ll.preamble_bits(0x8E89BED7)) == [1, 0, 1, 0, 1, 0, 1, 0]


@pytest.mark.parametrize("n", [0, 1, 2, 9, 37, 255])
def test_crc24_two_independent_formulations(n):
    data = os.urandom(n)
    assert ll.crc24_register(ll.bytes_to_bits(data)) == ll.crc24_bytes_fast(data)


def test_crc24_detects_all_single_and_double_bit_errors():
    pdu = os.urandom(39)
    bits = ll.bytes_to_bits(pdu)
    good = ll.crc24_register(bits)
    for i in range(len(bits)):
        b = bits.copy()
        b[i] ^= 1
        assert ll.crc24_register(b) != good
    rng = np.random.default_rng(3)
    for _ in range(300):
        b = bits.copy()
        i, j = rng.choice(len(bits), 2, replace=False)
        b[i] ^= 1
        b[j] ^= 1
        assert ll.crc24_register(b) != good


def test_crc24_is_linear_lfsr():
    # CRC(a) ^ CRC(b) ^ CRC(0) == CRC(a ^ b) for an affine LFSR with a preset
    a, b = os.urandom(20), os.urandom(20)
    x = bytes(p ^ q for p, q in zip(a, b))
    c = ll.crc24_bytes_fast
    assert c(a) ^ c(b) ^ c(bytes(20)) == c(x)


def test_whitening_initial_state_and_period():
    # channel 37 = 0b100101 -> positions 1..6 = 1,0,0,1,0,1 and position 0 = 1; first output = position 6
    seq = ll.whitening_sequence(37, 254)
    assert seq[0] == 1
    assert np.array_equal(seq[:127], seq[127:254])  # maximal-length 7-bit LFSR: period 127
    assert not np.array_equal(ll.whitening_sequence(37, 64), ll.whitening_sequence(38, 64))
    bits = np.random.default_rng(1).integers(0, 2, 300).astype(np.uint8)
    assert np.array_equal(ll.whiten(ll.whiten(bits, 39), 39), bits)


def test_channel_frequencies():
    assert ll.channel_freq(37) == 2402e6 and ll.channel_freq(38) == 2426e6 and ll.channel_freq(39) == 2480e6
    assert ll.channel_freq(0) == 2404e6 and ll.channel_freq(10) == 2424e6 and ll.channel_freq(11) == 2428e6
    assert ll.channel_freq(36) == 2478e6
    with pytest.raises(ValueError):
        ll.channel_freq(40)


# ---------------------------------------------------------------- PDU
def test_pdu_roundtrip_and_address_order():
    addr = bytes.fromhex("c1223344556f")
    p = adv_nonconn(addr, b"\x02\x01\x06")
    raw = p.encode()
    assert raw[0] == ADV_NONCONN_IND | 0x40 and raw[1] == 9
    assert raw[2:8] == addr[::-1]
    q = AdvPdu.decode(raw)
    assert q.adv_address == addr and q.adv_data == b"\x02\x01\x06" and q.tx_add and q.name == "ADV_NONCONN_IND"
    with pytest.raises(PduError):
        AdvPdu.decode(raw[:-1])
    with pytest.raises(PduError):
        adv_nonconn(addr, bytes(32))


def test_parse_ad():
    ads = parse_ad(bytes([2, 1, 6, 4, 9]) + b"abc" + bytes([0, 0]))
    assert [(a.ad_type, a.data) for a in ads] == [(1, b"\x06"), (9, b"abc")]
    with pytest.raises(PduError):
        parse_ad(bytes([5, 9, 1]))


# ---------------------------------------------------------------- modulator
def test_gfsk_phase_and_deviation():
    sps = 8
    bits = np.array([1] * 20 + [0] * 20 + [1, 0] * 20, dtype=np.uint8)
    x = modulate_bits(bits, sps=sps, guard_symbols=0)
    assert np.allclose(np.abs(x), 1.0)
    ph = np.unwrap(np.angle(x))
    # total phase change = pi h (#ones - #zeros) = pi/2 * 0
    assert ph[-1] - ph[0] == pytest.approx(0.0, abs=0.3)
    # long runs reach the nominal +-250 kHz deviation
    d = discriminate(x, sps * 1e6)
    assert d[10 * sps] == pytest.approx(1.0, abs=0.01) and d[30 * sps] == pytest.approx(-1.0, abs=0.01)
    # alternating 1010 pattern does not reach full deviation (Gaussian ISI, BT = 0.5)
    alt = np.abs(d[45 * sps:75 * sps]).max()
    assert 0.6 < alt < 0.95


def test_gfsk_occupied_bandwidth():
    rng = np.random.default_rng(0)
    x = modulate_bits(rng.integers(0, 2, 4000), sps=8, guard_symbols=0)
    X = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(len(x), 1 / 8e6))
    inside = X[np.abs(f) < 1e6].sum() / X.sum()
    assert inside > 0.99  # 99 % power bandwidth < 2 MHz (one BLE channel)


def test_frequency_waveform_centring():
    f = frequency_waveform(np.array([0] * 10 + [1] + [0] * 10), 16)
    assert int(np.argmax(f)) in (10 * 16 + 7, 10 * 16 + 8)  # bit 10's pulse centred on its symbol


# ---------------------------------------------------------------- receiver
def _advert():
    return adv_nonconn(bytes.fromhex("d0aabbccddee"), bytes([0x1E, 0xFF, 0xFF, 0xFF]) + os.urandom(27))


@pytest.mark.parametrize("fs", [2e6, 4e6, 8e6])
def test_receiver_noiseless(fs):
    pdu = _advert()
    x = np.concatenate([np.zeros(300), modulate_pdu(pdu, 37, sps=int(fs / 1e6)), np.zeros(300)])
    pk = BleReceiver(fs).decode(x)
    assert len(pk) == 1 and pk[0].crc_ok and pk[0].pdu == pdu.encode()


@pytest.mark.parametrize("cfo,delay,ebn0", [(150e3, 0.37, 16), (-150e3, 2.71, 16), (60e3, 1.5, 14)])
def test_receiver_cfo_timing_noise(cfo, delay, ebn0):
    rng = np.random.default_rng(int(abs(cfo)) + int(delay * 100))
    pdu = _advert()
    ok = 0
    for _ in range(10):
        x = modulate_pdu(pdu, 38, sps=4, guard_symbols=0)
        y = ch.apply(x, 4e6, rng, cfo_hz=cfo, delay=delay, lead=500, tail=200,
                     sigma2=ch.sigma2_for_ebn0(ebn0, 4))
        pk = BleReceiver(4e6, channel=38).decode(y)
        if pk and pk[0].pdu == pdu.encode():
            ok += 1
            assert pk[0].cfo_hz == pytest.approx(cfo, abs=15e3)
    assert ok >= 8


def test_receiver_wrong_channel_whitening_fails_crc():
    pdu = _advert()
    x = modulate_pdu(pdu, 37, sps=4)
    assert BleReceiver(4e6, channel=39).decode(x) == []
    pk = BleReceiver(4e6, channel=39, require_crc=False).decode(x)
    assert pk == [] or not pk[0].crc_ok


def test_receiver_rejects_noise_only():
    rng = np.random.default_rng(7)
    y = 0.1 * (rng.standard_normal(400000) + 1j * rng.standard_normal(400000))
    assert BleReceiver(4e6).decode(y) == []


def test_receiver_multiple_packets_and_multipath():
    rng = np.random.default_rng(11)
    pdus = [adv_nonconn(os.urandom(6), os.urandom(20)) for _ in range(5)]
    parts = []
    for p in pdus:
        parts += [np.zeros(int(rng.integers(100, 2000))), modulate_pdu(p, 37, sps=4, guard_symbols=0)]
    x = np.concatenate(parts + [np.zeros(500)])
    taps = ([0.0, 1.0, 2.5], [0.85, 0.45j, -0.25])
    y = ch.apply(x, 4e6, rng, cfo_hz=40e3, taps=taps, sigma2=ch.sigma2_for_ebn0(20, 4))
    got = [p.pdu for p in BleReceiver(4e6).decode(y)]
    assert got == [p.encode() for p in pdus]


def test_gardner_tracks_clock_offset():
    """2000 bits with a 300 ppm sample-clock offset: open-loop timing drifts
    0.6 symbol over the frame and fails; the Gardner loop tracks it."""
    rng = np.random.default_rng(5)
    bits = rng.integers(0, 2, 2000).astype(np.uint8)
    x = modulate_bits(np.concatenate([ll.sync_bits(), bits]), sps=4, guard_symbols=10)
    y = resample_ppm(x, 300.0)
    rx = BleReceiver(4e6)
    got = rx.raw_bits_after_sync(y, 2000)
    assert np.sum(got != bits) == 0
    # open loop (gains zero) fails
    d = rx._prepare(y)
    t_sync = max(rx.find_sync(d), key=lambda c: c[1])[0]
    gt, _ = rx.demod_from(y, t_sync, d)
    gt.kp = gt.ki = 0.0
    soft = gt.run(40 + 2000)
    open_loop = (np.array(soft[40:]) > 0).astype(np.uint8)
    assert np.sum(open_loop != bits) > 100


def test_gardner_converges_from_offset():
    sps = 4
    bits = np.random.default_rng(2).integers(0, 2, 400).astype(np.uint8)
    f = frequency_waveform(bits, sps)
    centre = (sps - 1) / 2  # samples k*sps .. k*sps+sps-1 carry bit k
    gt = GardnerTiming(f, centre + 1.3, sps)  # start 1.3 samples late
    gt.run(390)
    err = np.array(gt.strobes[200:390]) - (np.arange(200, 390) * sps + centre)
    assert np.max(np.abs(err)) < 0.1


def test_ber_matches_noncoherent_bound_at_8db():
    """Monte Carlo smoke test: the measured BER at Eb/N0 = 8 dB lies between the
    coherent MSK bound and 2x the non-coherent orthogonal BFSK bound."""
    from sdrmesh.montecarlo import ble_ber_point
    from sdrmesh.theory import ber_bfsk_noncoherent, ber_msk_coherent
    r = ble_ber_point(8.0, 30, 1000, seed=1)
    assert ber_msk_coherent(8.0) < r["ber"] < 2 * ber_bfsk_noncoherent(8.0)
