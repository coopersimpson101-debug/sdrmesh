# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-26

### Added
- BLE LE 1M PHY: GFSK modulator (BT 0.5, h 0.5), channel selection, FM discriminator with integrate-and-dump,
  sync-word correlation, CFO estimation and correction, Gardner timing recovery with a Farrow interpolator,
  de-whitening, CRC-24 and advertising-PDU parsing.
- LoRa PHY: chirp modulator/demodulator (SF7-SF12, any bandwidth), preamble detection, sync word, integer and
  fractional CFO, integer and fractional STO, Gray mapping, diagonal interleaver, Hamming FEC (CR 4/5-4/8),
  whitening, explicit/implicit header, CRC-16, low-data-rate optimisation, time on air.
- Channel model (AWGN, CFO, fractional delay, sample-clock offset, multipath) and Monte Carlo BER/SER/PER
  campaigns compared with analytic curves.
- SDR I/O: cs8/cs16/cf32 files, SigMF metadata, HackRF (`hackrf_transfer`) and PlutoSDR (pyadi-iio) sources,
  ring buffer and a streaming receiver; CLI `sdrmesh rx ble --source hackrf|pluto|file|stdin`.
- meshfind integration: adverts built with the meshfind reference code decode end to end; meshfind report frames
  are carried over a LoRa router-to-router frame format, with airtime and throughput tables.

### Not yet verified
- Any over-the-air reception or transmission (no SDR was available); see the README hardware section.
