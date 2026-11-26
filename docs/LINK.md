# Long-range link prototype: meshfind reports over LoRa

meshfind relays forward encrypted location reports over ESP-NOW (2.4 GHz, tens of metres per hop). To join
sites that are hundreds of metres to kilometres apart, a *router* node can bridge meshfind mesh frames over a
LoRa link. This document specifies the frame format implemented in `sdrmesh/link.py` and the airtime and
throughput it allows. Everything here is computed or simulated; nothing has been measured over the air yet.

## Why no extra crypto

A meshfind report is already ECIES-sealed to the tag owner (X25519 + ChaCha20-Poly1305, 108 bytes) and
indexed by `SHA-256(P_i)`. A router learns nothing it could not learn from the ESP-NOW mesh, so the LoRa link
only needs integrity against noise, which the LoRa PHY CRC-16 provides. Authenticating routers to each other
(to stop a rogue node from flooding the link) is listed in the roadmap.

## Frame format (LR v1)

```
0  u8    magic 0x4C ('L')
1  u8    version (4 bits, = 1) << 4 | type (1 = DATA, 2 = ACK)
2  u16   source router id          (little-endian)
4  u16   destination router id     (0xFFFF = broadcast)
6  u8    sequence number
7  u8    item count n
8  ...   DATA: n x (u8 length | meshfind mesh frame)      ACK: n x u8 acknowledged sequence numbers
```

A meshfind REPORT mesh frame is 12 + 108 = 120 bytes (`meshfind.mesh.Frame`), so a LoRa packet (255-byte
payload limit) carries two of them: 8 + 2 x 121 = 250 bytes. `pack_reports()` bundles greedily;
`LRFrame.decode()` rejects truncated items, trailing bytes, unknown types and wrong magic/version.

Suggested PHY settings: explicit header, CRC on, sync word 0x12 (private network, not LoRaWAN), preamble 8.

## Airtime and throughput

Time on air uses the Semtech formula (`LoRaParams.airtime`), which the modulator reproduces exactly
(`test_modulated_length_equals_airtime`). Under the EU 868 MHz 1 % duty-cycle rule a router may transmit
36 s per hour. `sdrmesh link-budget` prints the table for any coding rate; the full table, with the simulated
SNR thresholds of this repository's own receiver and the resulting sensitivity estimate
(`-174 dBm/Hz + 10 log10(BW) + NF + SNR_req`, NF = 6 dB assumed) and path-loss budget at +14 dBm, is in
[results/SUMMARY.md](results/SUMMARY.md).

Reading the table: at SF7/125 kHz one 250-byte packet takes about 0.39 s, allowing about 185 reports per hour at
1 % duty cycle; SF12 buys about 14 dB more link budget (simulated thresholds) but allows only about 8 reports per hour. A
meshfind relay produces one report per tag sighting, rate-limited per key, so a router serving a few tags can
use SF9 to SF10; a busy site needs SF7 or a 10 % duty-cycle sub-band (869.4 to 869.65 MHz, 500 mW ERP).

## Tested end to end

`tests/test_meshfind_e2e.py::test_meshfind_report_over_lora_link_owner_decrypts`: meshfind seals a report for a
tag key, wraps it in a meshfind mesh frame, `pack_reports` builds an LR frame, the LoRa PHY (SF9, CR 4/8, 2x
oversampled) carries it through a channel with 7.3 kHz CFO, a 3.3-sample delay and -9 dB SNR, and the owner
decrypts both reports with meshfind's `report.open_`.
