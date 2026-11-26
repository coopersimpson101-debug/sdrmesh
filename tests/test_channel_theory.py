import math

import numpy as np
import pytest

from sdrmesh import channel as ch
from sdrmesh import theory as th


def test_awgn_calibration():
    rng = np.random.default_rng(0)
    s2 = ch.sigma2_for_ebn0(10.0, sps=4)
    assert s2 == pytest.approx(0.4)
    n = ch.awgn(np.zeros(200000, dtype=complex), s2, rng)
    assert np.var(n) == pytest.approx(s2, rel=0.02)
    assert np.var(n.real) == pytest.approx(s2 / 2, rel=0.02)
    # SNR over BW at fs = 4 BW: noise variance per sample is 4x the in-band noise
    assert ch.sigma2_for_snr(0.0, 4e6, 1e6) == pytest.approx(4.0)


def test_cfo_and_phase():
    rng = np.random.default_rng(1)
    x = np.ones(1000, dtype=complex)
    y = ch.apply(x, 1e6, rng, cfo_hz=1e3, phase=0.0)
    assert np.angle(y[250]) == pytest.approx(2 * math.pi * 1e3 * 250 / 1e6)


def test_multipath_taps():
    x = np.zeros(64, dtype=complex)
    x[10] = 1.0
    y = ch.multipath(x, [0.0, 3.0], [1.0, 0.5j], bandlimited=False)
    assert y[10] == pytest.approx(1.0) and y[13] == pytest.approx(0.5j)
    d, g = ch.rayleigh_taps(4, 1.0, 3.0, np.random.default_rng(0))
    assert d == [0.0, 1.0, 2.0, 3.0] and len(g) == 4


def test_noncoherent_mfsk_integral_matches_closed_form_for_binary():
    for e in (0.5, 2.0, 8.0, 20.0):
        assert th.ser_mfsk_noncoherent(2, e) == pytest.approx(0.5 * math.exp(-e / 2), rel=1e-4)


def test_noncoherent_mfsk_matches_binomial_sum_small_n():
    # exact alternating sum, fine for small N
    for n in (4, 8, 16):
        for e in (1.0, 5.0, 12.0):
            ps = sum((-1) ** (k + 1) * math.comb(n - 1, k) / (k + 1) * math.exp(-k / (k + 1) * e)
                     for k in range(1, n))
            assert th.ser_mfsk_noncoherent(n, e) == pytest.approx(ps, rel=1e-5, abs=1e-12)


def test_lora_theory_monotone_and_sf_gain():
    snr = np.arange(-25, -2, 1.0)
    for sf in (7, 12):
        s = th.lora_ser(sf, snr)
        assert np.all(np.diff(s) <= 1e-12)

    # each SF step buys about 2.5 dB at SER 1e-3 (the well-known LoRa rule of thumb)
    def thr(sf):
        centre = -8.5 - 2.5 * (sf - 7)
        grid = np.arange(centre - 3, centre + 3, 0.05)
        return grid[np.argmax(th.lora_ser(sf, grid) < 1e-3)]
    steps = [thr(sf) - thr(sf + 1) for sf in range(7, 12)]
    assert all(2.0 < s < 3.2 for s in steps)


def test_bfsk_curves():
    assert th.ber_bfsk_noncoherent(0.0) == pytest.approx(0.5 * math.exp(-0.5))
    assert th.ber_msk_coherent(9.6) == pytest.approx(1e-5, rel=0.1)
    assert th.lora_snr_to_ebn0(7, 0.0) == pytest.approx(10 * math.log10(128 / 7))
