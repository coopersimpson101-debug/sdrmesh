import math

import numpy as np
import pytest

from sdrmesh.dsp import (channel_select, cubic_at, decimate, filt, fractional_delay, fractional_delay_fft, freq_shift,
                         gaussian_pulse, lowpass, resample_ppm, spectrogram, window)


def test_windows_symmetric_and_bounded():
    for name in ("hann", "hamming", "blackman"):
        w = window(name, 65)
        assert np.allclose(w, w[::-1])
        assert w.max() <= 1.0 + 1e-12 and w[32] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        window("kaiser", 8)


def test_lowpass_response():
    fs = 1.0
    h = lowpass(101, 0.1, fs)
    assert h.sum() == pytest.approx(1.0)
    H = np.abs(np.fft.rfft(h, 8192))
    f = np.fft.rfftfreq(8192)
    assert np.all(H[f < 0.07] > 0.99)
    assert H[np.argmin(abs(f - 0.1))] == pytest.approx(0.5, abs=0.02)  # -6 dB edge
    assert np.all(H[f > 0.14] < 1e-3)  # Blackman: > 60 dB stop band


def test_filt_fft_and_direct_paths_agree():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(5000) + 1j * rng.standard_normal(5000)
    h = lowpass(129, 0.2, 1.0)
    direct = np.convolve(x, h)[64:64 + 5000]
    assert np.allclose(filt(x, h), direct)


def test_gaussian_pulse_bt():
    g = gaussian_pulse(0.5, 64, span=3)
    assert g.sum() == pytest.approx(1.0)
    # 3 dB bandwidth of the Gaussian filter equals B = BT / T
    G = np.abs(np.fft.rfft(g, 1 << 16))
    f = np.fft.rfftfreq(1 << 16, d=1 / 64)  # in units of 1/T
    f3 = f[np.argmin(abs(G / G[0] - 1 / math.sqrt(2)))]
    assert f3 == pytest.approx(0.5, rel=0.02)


def test_cubic_interpolation_exact_for_cubics():
    n = np.arange(20.0)
    x = 0.3 * n ** 3 - 2 * n ** 2 + n - 4
    t = np.array([3.25, 7.5, 11.9])
    assert np.allclose(cubic_at(x, t), 0.3 * t ** 3 - 2 * t ** 2 + t - 4)


def test_fractional_delays():
    n = np.arange(400)
    x = np.exp(2j * math.pi * 0.05 * n)
    for d in (0.3, 2.7):
        ref = np.exp(2j * math.pi * 0.05 * (n - d))
        assert np.max(abs(fractional_delay(x, d)[10:-10] - ref[10:-10])) < 2e-3
        assert np.max(abs(fractional_delay_fft(x, d)[30:-30] - ref[30:-30])) < 2e-2


def test_resample_ppm_tone_frequency():
    n = np.arange(200000)
    x = np.exp(2j * math.pi * 0.01 * n)
    y = resample_ppm(x, 1000.0)  # receiver clock 1000 ppm fast -> tone appears lower
    ph = np.unwrap(np.angle(y))
    f = (ph[-1000] - ph[1000]) / (2 * math.pi * (len(y) - 2000))
    assert f == pytest.approx(0.01 * (1 - 1e-3), rel=1e-5)


def test_channel_select_and_decimate():
    fs = 8e6
    n = np.arange(80000)
    tone = np.exp(2j * math.pi * (1.25e6) / fs * n)  # 1 MHz channel offset + 250 kHz
    blocker = np.exp(2j * math.pi * (-2.5e6) / fs * n)
    y = channel_select(tone + blocker, fs, 1e6, 4e6, 2e6)
    f, = np.fft.fftfreq(len(y), 1 / 4e6)[[np.argmax(abs(np.fft.fft(y)))]],
    assert f == pytest.approx(250e3, abs=100)
    spec = np.abs(np.fft.fft(y[200:-200])) ** 2
    assert spec.max() / np.sort(spec)[-3] > 1e4  # blocker gone
    z = decimate(np.ones(1000), 4)
    assert np.allclose(z[20:-20], 1.0, atol=1e-6)


def test_freq_shift_phase_continuity():
    x = np.ones(100, dtype=complex)
    a = freq_shift(x, 0.1, 1.0)
    b = np.concatenate([freq_shift(x[:37], 0.1, 1.0), freq_shift(x[37:], 0.1, 1.0, n0=37)])
    assert np.allclose(a, b)


def test_spectrogram_tone_peak():
    fs = 1e6
    n = np.arange(20000)
    x = np.exp(2j * math.pi * 125e3 / fs * n)
    t, f, s = spectrogram(x, fs, nfft=256)
    assert f[np.argmax(s.mean(axis=0))] == pytest.approx(125e3, abs=fs / 256)
    assert s.max() == pytest.approx(0.0, abs=0.1)  # unit-power tone reads 0 dB
