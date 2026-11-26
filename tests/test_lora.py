import itertools
import math

import numpy as np
import pytest

from sdrmesh import channel as ch
from sdrmesh.lora import coding as c
from sdrmesh.lora.chirp import demod_symbols, modulate_symbols, upchirp
from sdrmesh.lora.modem import LoRaReceiver, modulate, n_samples, sync_symbols
from sdrmesh.lora.params import LoRaParams

# First 32 entries of the whitening table published in gr-lora_sdr (lib/tables.h);
# our LFSR must reproduce them.
GR_LORA_SDR_WHITENING_HEAD = bytes.fromhex(
    "fffefcf8f0e1c2850b172f5ebc78f1e3c68d1a3468d0a0408001020408112347")


def test_whitening_matches_published_table():
    assert c.whitening_sequence(32) == GR_LORA_SDR_WHITENING_HEAD
    seq = c.whitening_sequence(510)
    assert seq[:255] == seq[255:]  # maximal-length 8-bit LFSR, period 255
    assert len(set(seq[:255])) == 255 and 0 not in seq[:255]


def test_header_checksum_roundtrip_and_detection():
    for ln, cr, crc in itertools.product([0, 1, 37, 128, 255], [1, 2, 3, 4], [True, False]):
        nib = c.header_nibbles(ln, cr, crc)
        assert c.parse_header(nib) == (ln, cr, crc)
        for i in range(3):  # any single flipped bit in the protected nibbles is caught
            for b in range(4):
                bad = list(nib)
                bad[i] ^= 1 << b
                with pytest.raises(c.LoRaDecodeError):
                    c.parse_header(bad)


def test_crc16_ccitt_reference():
    # CRC-16/XMODEM (poly 0x1021, init 0) check value for "123456789" is 0x31C3
    assert c.crc16_ccitt(b"123456789") == 0x31C3


@pytest.mark.parametrize("cr", [1, 2, 3, 4])
def test_hamming_codebook_and_correction(cr):
    book = [c.hamming_encode(v, cr) for v in range(16)]
    n = 4 + cr
    assert all(0 <= w < (1 << n) for w in book)
    dmin = min(bin(a ^ b).count("1") for a, b in itertools.combinations(book, 2))
    assert dmin == {1: 2, 2: 2, 3: 3, 4: 4}[cr]
    for v in range(16):
        assert c.hamming_decode(book[v], cr) == (v, False)
        for b in range(n):
            got, flag = c.hamming_decode(book[v] ^ (1 << b), cr)
            assert flag
            if cr >= 3:
                assert got == v  # single errors corrected for 4/7 and 4/8


@pytest.mark.parametrize("sf", [7, 8, 12])
def test_interleaver_inverse(sf):
    rng = np.random.default_rng(sf)
    for sf_app, cw_len in ((sf - 2, 8), (sf, 5), (sf, 8)):
        cws = [int(v) for v in rng.integers(0, 1 << cw_len, sf_app)]
        syms = c.interleave_block(cws, sf_app, cw_len, sf)
        assert len(syms) == cw_len and all(0 <= s < (1 << sf) for s in syms)
        words = [s >> 2 for s in syms] if sf_app == sf - 2 else syms
        assert c.deinterleave_block(words, sf_app, cw_len) == cws


def test_interleaver_spreads_burst():
    """One corrupted symbol damages at most one bit of each codeword (so 4/7,
    4/8 can correct a whole lost symbol)."""
    sf, cw_len = 9, 8
    cws = [0] * sf
    syms = c.interleave_block(cws, sf, cw_len, sf)
    syms[3] ^= (1 << sf) - 1
    out = c.deinterleave_block(syms, sf, cw_len)
    assert all(bin(w).count("1") <= 1 for w in out)


def test_gray():
    for v in range(4096):
        assert c.gray_inverse(c.gray(v)) == v
        assert bin(c.gray(v) ^ c.gray(v + 1)).count("1") == 1


@pytest.mark.parametrize("sf,cr,ldro,implicit", [(7, 1, False, False), (8, 4, False, False), (12, 2, True, False),
                                                 (9, 3, False, True), (11, 1, True, True)])
def test_frame_codec_roundtrip(sf, cr, ldro, implicit):
    rng = np.random.default_rng(sf * 10 + cr)
    for ln in (1, 2, 5, 17, 64, 255):
        pl = bytes(rng.integers(0, 256, ln, dtype=np.uint8))
        syms = c.encode_frame(pl, sf, cr, True, implicit, ldro)
        p = LoRaParams(sf=sf, cr=cr, implicit_header=implicit, ldro=ldro)
        assert len(syms) == p.payload_symbols(ln)  # matches Semtech's symbol count formula
        r = c.decode_frame(syms, sf, cr=cr if implicit else None, has_crc=True if implicit else None,
                           implicit_len=ln if implicit else None, ldro=ldro)
        assert r["payload"] == pl and r["crc_ok"] and r["fec_errors"] == 0


def test_frame_codec_fec_corrects_symbol_error():
    pl = b"meshfind over the long link"
    syms = c.encode_frame(pl, 8, 4)
    syms[12] = (syms[12] + 37) % 256  # one wrong chirp in the payload
    r = c.decode_frame(syms, 8)
    assert r["payload"] == pl and r["crc_ok"] and r["fec_errors"] > 0
    syms = c.encode_frame(pl, 8, 1)
    # symbol 10 is a data row of the first CR 4/5 block: 4/5 cannot correct it, the CRC must catch it
    syms[10] = (syms[10] + 37) % 256
    assert not c.decode_frame(syms, 8)["crc_ok"]
    # symbol 12 is that block's parity row: only parity bits are hit, the data survive
    syms = c.encode_frame(pl, 8, 1)
    syms[12] = (syms[12] + 37) % 256
    r = c.decode_frame(syms, 8)
    assert r["crc_ok"] and r["fec_errors"] > 0


def test_header_block_reduced_rate_tolerates_one_bin():
    """The first block drops the two LSBs of each symbol: +-1 bin errors do not
    corrupt the header."""
    syms = c.encode_frame(b"abc", 9, 2)
    for k in range(8):
        bad = list(syms)
        bad[k] = (bad[k] + 1) % 512
        assert c.decode_frame(bad, 9)["payload"] == b"abc"


# ---------------------------------------------------------------- chirps
@pytest.mark.parametrize("sf", [7, 10, 12])
def test_chirps_orthogonal_and_demodulate(sf):
    N = 1 << sf
    s = np.random.default_rng(sf).integers(0, N, 50)
    assert np.array_equal(demod_symbols(modulate_symbols(s, sf), sf), s)
    a, b = upchirp(sf, 3), upchirp(sf, 77)
    assert abs(np.vdot(a, b)) / N < 1e-9
    assert np.allclose(np.abs(a), 1)


def test_chirp_instantaneous_frequency():
    sf, os_ = 7, 8
    x = upchirp(sf, 32, os_)
    f = np.angle(x[1:] * np.conj(x[:-1])) / (2 * math.pi) * os_  # in units of BW
    assert f[0] == pytest.approx(32 / 128 - 0.5, abs=0.01)
    assert np.all(f <= 0.5 + 1e-6) and np.all(f >= -0.5 - 1e-6)


def test_sync_symbols():
    assert sync_symbols(0x12) == (8, 16)
    assert sync_symbols(0x34) == (24, 32)


# ---------------------------------------------------------------- params / airtime
def test_airtime_formula_values():
    # SF7/125k/4/5, 13 B, CRC on, explicit header, 8-symbol preamble:
    # T_sym = 1.024 ms; payload symbols 8 + ceil((104-28+28+16)/28)*5 = 33 -> (12.25 + 33) * 1.024 ms
    p = LoRaParams(sf=7)
    assert p.payload_symbols(13) == 33
    assert p.airtime(13) == pytest.approx(46.336e-3)
    # SF12/125k: LDRO mandatory (T_sym = 32.768 ms > 16 ms)
    p12 = LoRaParams(sf=12)
    assert p12.low_data_rate and not LoRaParams(sf=12, bw=500e3).low_data_rate
    assert p12.payload_symbols(13) == 8 + math.ceil((104 - 48 + 28 + 16) / 40) * 5


@pytest.mark.parametrize("sf,bw", [(7, 125e3), (9, 250e3), (12, 500e3), (11, 125e3)])
def test_modulated_length_equals_airtime(sf, bw):
    p = LoRaParams(sf=sf, bw=bw, cr=3, os=2)
    for ln in (1, 20, 100):
        x = modulate(bytes(ln), p, pad_symbols=0)
        assert len(x) == n_samples(ln, p)


# ---------------------------------------------------------------- full modem
SF_BW = [(sf, bw) for sf in range(7, 13) for bw in (125e3, 250e3, 500e3)]


@pytest.mark.parametrize("sf,bw", SF_BW)
def test_modem_all_sf_bw_noiseless(sf, bw):
    rng = np.random.default_rng(sf)
    for cr in (1, 4):
        p = LoRaParams(sf=sf, bw=bw, cr=cr)
        pl = bytes(rng.integers(0, 256, 12, dtype=np.uint8))
        y = ch.apply(modulate(pl, p, pad_symbols=0), p.fs, rng, lead=int(1.7 * p.n), tail=3 * p.n)
        fr = LoRaReceiver(p).decode(y)
        assert len(fr) == 1 and fr[0].payload == pl and fr[0].crc_ok and fr[0].cr == cr


@pytest.mark.parametrize("sf", [7, 9, 12])
def test_modem_cfo_sto_noise(sf):
    """Integer + fractional CFO, fractional timing at 4x oversampling and
    in-band SNR 3 dB above the uncoded threshold region."""
    rng = np.random.default_rng(100 + sf)
    snr = {7: -3.0, 9: -8.0, 12: -15.0}[sf]
    p = LoRaParams(sf=sf, cr=2, os=4)
    ok = 0
    for _ in range(4):
        pl = bytes(rng.integers(0, 256, 16, dtype=np.uint8))
        cfo_bins = float(rng.uniform(-30, 30))
        y = ch.apply(modulate(pl, p, pad_symbols=0), p.fs, rng, cfo_hz=cfo_bins * p.bw / p.n,
                     delay=float(rng.uniform(0, 8)), lead=int(rng.integers(0, 3 * p.n * 4)), tail=3 * p.n * 4,
                     sigma2=ch.sigma2_for_snr(snr, p.fs, p.bw))
        fr = LoRaReceiver(p).decode(y)
        if fr and fr[0].payload == pl and fr[0].crc_ok:
            ok += 1
            assert fr[0].cfo_hz == pytest.approx(cfo_bins * p.bw / p.n, abs=0.1 * p.bw / p.n)
    assert ok == 4


def test_modem_two_frames_and_lorawan_sync_word():
    rng = np.random.default_rng(3)
    p = LoRaParams(sf=8, cr=1, sync_word=0x34)
    a, b = b"first", b"second frame"
    x = np.concatenate([modulate(a, p), np.zeros(1000), modulate(b, p)])
    got = [f.payload for f in LoRaReceiver(p).decode(ch.apply(x, p.fs, rng, sigma2=ch.sigma2_for_snr(0, p.fs, p.bw)))]
    assert got == [a, b]
    # a receiver expecting the private sync word 0x12 must not lock onto 0x34 frames
    assert LoRaReceiver(LoRaParams(sf=8, cr=1, sync_word=0x12)).decode(x) == []


def test_modem_implicit_header():
    p = LoRaParams(sf=9, cr=3, implicit_header=True)
    x = modulate(b"implicit!", p)
    fr = LoRaReceiver(p).decode(x, implicit_len=9)
    assert fr and fr[0].payload == b"implicit!" and fr[0].crc_ok


def test_lora_uncoded_ser_matches_theory():
    from sdrmesh.montecarlo import lora_ser_point
    from sdrmesh.theory import lora_ser
    r = lora_ser_point(7, -10.0, 6000, seed=2)
    th = float(lora_ser(7, -10.0)[0])
    sd = math.sqrt(th * (1 - th) / 6000)
    assert abs(r["ser"] - th) < 4 * sd + 1e-3
