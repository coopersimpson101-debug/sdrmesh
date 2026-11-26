# PHY notes: derivations, conventions and what is (not) validated

This file holds the longer derivations behind the README. Code references are to `sdrmesh/`.

## 1. BLE LE 1M

### 1.1 Modulation (`ble/gfsk.py`)

LE 1M is binary continuous-phase FSK at 1 Msym/s with Gaussian pulse shaping (BT = 0.5) and modulation index
h = 0.5 (allowed 0.45 to 0.55). With NRZ symbols `a_k = 2 b_k - 1` and the frequency pulse
`g(t) = rect_T(t) * h_G(t)`, where `h_G(t) = sqrt(2 pi / ln 2) B exp(-2 pi^2 B^2 t^2 / ln 2)` and `B = BT / T`:

```
phi(t) = pi h sum_k a_k q(t - kT),     q(t) = (1/T) int_{-inf}^{t} g(s) ds,   q(inf) = 1
f(t)   = (h / 2T) sum_k a_k g(t - kT)  ->  +-250 kHz for long runs
```

The implementation samples `g` at `sps` samples per symbol (span 3 T), filters the up-sampled NRZ sequence
and integrates the phase with a cumulative sum. Two carrier-only symbols are added before and after the packet
(PA ramp), so the receiver's filters see a carrier, not silence, around the first and last bits. Tests check
constant envelope, the +-250 kHz deviation, the reduced swing on `1010` patterns (Gaussian ISI) and a 99 %
power bandwidth inside 2 MHz.

### 1.2 Receiver front end (`ble/receiver.py`)

* **Channel filter**: 33-tap (at 4 MS/s) Blackman windowed-sinc, 600 kHz -6 dB edge. A sweep on this machine
  (500 / 600 / 800 kHz, 33 / 65 taps) put the BER minimum at 600 kHz.
* **Discriminator**: `d[n] = arg(r[n] r*[n-1]) fs / (2 pi f_dev)`. For the CPFSK signal plus CFO `f_o` this is
  `f(t)/f_dev + f_o/f_dev` plus noise (with "click" noise at low SNR).
* **Integrate and dump**: a moving average over one symbol before sampling. It is the matched filter for the
  rectangular part of the frequency pulse; it moved the measured AWGN BER from about 5e-2 to 4e-3 at
  Eb/N0 = 10 dB, close to the non-coherent BFSK curve.

### 1.3 Synchronisation

The 40-bit sync word (preamble 0xAA + access address 0x8E89BED6, LSB first) is located by normalised
cross-correlation of `d` with a template `t` made by passing the noiseless sync word through the same front end:

```
rho[n] = sum_m d[n+m] t0[m] / ( ||t0|| sqrt( sum_m (d[n+m] - mu_n)^2 ) ),    t0 = t - mean(t)
```

Because `sum t0 = 0`, the numerator ignores any DC in `d`, so the CFO does not move the peak; the
denominator makes the metric gain-independent. The numerator is one FFT correlation and the denominator two
running sums (cumulative sums), so the cost is O(N log N) per block. Peaks above 0.55 are grouped, the maximum
is refined with a three-point parabola, and a cheap check slices the 32 access-address bits at nominal timing
before any expensive work (on 15 blocks of 65 536 pure-noise samples it rejected 42 of 50 false sync peaks
before the timing loop ran; the CRC rejects the rest).

**CFO**: the access address is not DC-balanced, so the estimator subtracts the template's own mean:
`f_o = f_dev (mean(d[n0 : n0+L]) - mean(t))`. The IQ segment is then de-rotated by `exp(-j 2 pi f_o n / fs)` and
re-filtered/re-discriminated, so the slicer threshold is zero. In the tests the estimate is within 15 kHz at
+-150 kHz offsets and Eb/N0 = 16 dB (the spec tolerance for a transmitter is +-150 kHz).

**Timing**: Gardner TED with a PI loop and a cubic Farrow interpolator (`ble/timing.py`). The front-end delay
from symbol start to eye centre is calibrated once at start-up on a noiseless random sequence. The loop
tracks a 300 ppm sample-clock offset over a 2000-bit frame with zero errors, where open-loop sampling makes
more than 100 errors (`test_gardner_tracks_clock_offset`).

### 1.4 Link layer (`ble/linklayer.py`, `ble/pdu.py`)

* Bit order: LSB first for every field except the CRC (MSB first, position 23 to 0).
* CRC-24: LFSR `x^24 + x^10 + x^9 + x^6 + x^4 + x^3 + x + 1`, preset 0x555555 on advertising channels (position
  0 = preset LSB). Implemented twice (bit-serial spec form and a byte-wise reflected table) and cross-checked;
  tests also check that it detects all single-bit and sampled double-bit errors and that it is affine.
* Whitening: `x^7 + x^4 + 1`, position 0 = 1, positions 1..6 = channel index (MSB in position 1); period 127.
* PDU header: type, ChSel, TxAdd, RxAdd, length; AdvA is little-endian on air.

**Validation status.** I could not find an openly licensed BLE IQ capture from an official or academic source
that I could fetch and cite here, so the BLE PHY has only been tested on signals synthesised by its own
modulator. The link-layer conventions follow the Core Specification text; the two independent CRC
formulations agree with each other, which shows internal consistency but does not by itself prove
interoperability. A capture from a real advertiser (see the README hardware section) is the outstanding test.

## 2. LoRa

### 2.1 Chirps (`lora/chirp.py`)

```
x_s[n] = exp(j 2 pi ( n^2 / (2 N os^2) + (s/N - 1/2) n / os ))   for n < (N - s) os,
         the same with -3/2 in place of -1/2 afterwards (the frequency wraps from +BW/2 to -BW/2),
N = 2^SF,  fs = os * BW.
```

Dechirping (multiplying by `x_0*`) leaves `exp(j 2 pi s n / N)`; the N symbols are orthogonal, so the ML
non-coherent detector is `argmax_k |DFT_N{x x_0*}[k]|`.

### 2.2 Frame synchronisation (`lora/modem.py`)

Let a window start `tau` samples after a symbol boundary and let the CFO be `c` bins (`c BW / N` Hz).

* Up-chirp, dechirped with the down-chirp: tone at bin `c + tau`.
* Down-chirp, dechirped with the up-chirp: tone at bin `c - tau`.

Steps:

1. Detection: four or more consecutive symbol-spaced windows whose peak bins agree to +-2.
2. Oversampled input: pick the decimation phase with the sharpest preamble peak.
3. Fractional CFO: the preamble is N-periodic, so a tone of `c` bins rotates by `2 pi c` from one window to
   the next whatever `tau` is: `c_frac = arg( sum_i sum_{k in peak +-1} X_i[k] X_{i-1}[k]* ) / 2 pi`.
4. Re-align the grid by the integer up-chirp bin `k_up` (now `tau = -c`), find the sync word (0x12 -> chirps 8
   and 16), then the down-chirps give `k_down = 2 c`, so `c_int = k_down / 2` (unambiguous for |c| < N/4).
5. Fractional STO: after removing the CFO the up-chirp peak sits at `-eps`; Jacobsen's estimator
   `delta = Re[(X[k-1] - X[k+1]) / (2 X[k] - X[k-1] - X[k+1])]` averaged over preamble windows.
6. Demodulate the 8 header symbols, decode the header, then exactly the number of symbols the header implies.

Not modelled: sampling-frequency offset (which on real LoRa radios is locked to the CFO and matters at SF11/12
for long frames), and soft-decision decoding.

### 2.3 Coding chain (`lora/coding.py`)

| stage | rule (as documented by Knight & Seeber 2016 and implemented in gr-lora_sdr) |
|---|---|
| whitening | payload bytes XOR the LFSR `x^8+x^6+x^5+x^4+1` sequence (seed 0xFF); header and CRC not whitened; nibbles low first |
| header | `len>>4, len&15, CR<<1 | CRC, c4, c3..c0`, 5-bit checksum from fixed XOR equations |
| CRC | CRC-16/CCITT (0x1021, init 0) over `payload[:-2]`, XOR `payload[-1] ^ payload[-2] << 8` |
| Hamming | codeword `d0 d1 d2 d3 p0 p1 p2 p3` truncated to 4+CR bits; 4/5 uses a single parity bit |
| interleaver | block of SF' codewords -> 4+CR symbols, symbol i bit j = bit i of codeword (i-j-1) mod SF' |
| reduced rate | first block (header) always CR 4/8 and SF' = SF-2 (+parity and 0 in the two LSBs); all blocks when LDRO |
| Gray | TX symbol = (gray^-1(word) + 1) mod N; RX word = gray((bin - 1) mod N >> 2 if reduced) |

Decoding: nearest-codeword (ML hard) decoding for 4/7 and 4/8 corrects any single error per codeword; 4/5
and 4/6 only detect. The interleaver spreads one corrupted chirp over at most one bit per codeword
(`test_interleaver_spreads_burst`), so 4/7 and 4/8 survive a whole wrong symbol.

**Validation status.** The gr-lora_sdr repository publishes no IQ test vectors that I could find, so the LoRa
PHY is **validated only against its own modulator**. The one published constant checked here is the head of
the whitening table in gr-lora_sdr's `lib/tables.h` (reproduced by the LFSR). Bit-compatibility with a real
SX127x/SX126x, or with gr-lora_sdr, is therefore unverified (see the README hardware section).

### 2.4 Theory used for comparison (`theory.py`)

Non-coherent orthogonal N-FSK with the correct bin Rician and N-1 bins Rayleigh:

```
P_c = int_0^inf r exp(-(r^2 + a^2)/2) I0(a r) (1 - exp(-r^2/2))^(N-1) dr,   a^2 = 2 Es/N0,  Es/N0 = N SNR
```

evaluated in the log domain (scaled Bessel function, `log1p`/`expm1`) on a dense grid, because the
alternating binomial closed form cancels catastrophically for N = 4096. Tests check it against the exact
binary formula and against the binomial sum for N = 4, 8, 16, and that each SF step buys 2.0 to 3.2 dB at
SER 1e-3.

## References

1. Bluetooth SIG, *Bluetooth Core Specification v5.4*, Vol 6 (Low Energy Controller), Parts A and B, 2023.
2. F. M. Gardner, "A BPSK/QPSK timing-error detector for sampled receivers", IEEE Trans. Commun. 34(5), 1986.
3. L. Erup, F. M. Gardner, R. A. Harris, "Interpolation in digital modems, part II", IEEE Trans. Commun. 41(6), 1993.
4. J. G. Proakis, M. Salehi, *Digital Communications*, 5th ed., McGraw-Hill, 2008.
5. M. Knight, B. Seeber, "Decoding LoRa: Realizing a Modern LPWAN with SDR", Proc. GNU Radio Conf., 2016.
6. J. Tapparel, O. Afisiadis, P. Mayoraz, A. Balatsoukas-Stimming, A. Burg, "An Open-Source LoRa Physical Layer
   Prototype on GNU Radio", IEEE SPAWC 2020; code: https://github.com/tapparelj/gr-lora_sdr
7. C. Bernier, F. Dehmas, N. Deparis, "Low complexity LoRa frame synchronization for ultra-low power
   software-defined radios", IEEE Trans. Commun. 68(5), 2020.
8. L. Vangelista, "Frequency shift chirp modulation: the LoRa modulation", IEEE Signal Process. Lett. 24(12), 2017.
9. T. Elshabrawy, J. Robert, "Closed-form approximation of LoRa modulation BER performance", IEEE Commun. Lett.
   22(9), 2018.
10. Semtech, *SX1276/77/78/79 datasheet*, rev. 7, 2020, section 4.1.1 (time on air); AN1200.13.
11. E. Jacobsen, P. Kootsookos, "Fast, accurate frequency estimators", IEEE Signal Process. Mag. 24(3), 2007.
12. SigMF specification v1.0.0, https://github.com/sigmf/SigMF
