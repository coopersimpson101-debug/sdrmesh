"""Command-line interface.

    sdrmesh rx ble --source hackrf|pluto|file|stdin [...]   decode BLE adverts (meshfind tags)
    sdrmesh tx ble --meshfind-demo -o capture.cs8           synthesise a test capture
    sdrmesh lora tx --text "hi" -o frame.cf32               LoRa modulator
    sdrmesh lora rx --file frame.cf32                       LoRa receiver
    sdrmesh airtime --sf 9 --bw 125e3 --len 250             time on air
    sdrmesh link-budget                                     meshfind-over-LoRa table
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from . import __version__


def _print_packet(p, as_json: bool, out=None) -> None:
    out = out or sys.stdout
    from .meshfind_bridge import HAVE_MESHFIND, decode_meshfind, key_hash_hex
    a = p.adv
    rec = {
        "t_ms": round(p.extra.get("time_s", 0.0) * 1e3, 3),
        "ch": p.channel, "pdu": a.name if a else "?", "len": len(p.pdu) - 2,
        "addr": a.adv_address.hex(":") if a and a.adv_address else None,
        "rssi_dbfs": round(p.rssi_db, 1), "cfo_khz": round(p.cfo_hz / 1e3, 1), "crc": "ok" if p.crc_ok else "BAD",
    }
    tag = decode_meshfind(p) if HAVE_MESHFIND else None
    if tag is not None:
        rec["meshfind"] = {"key_hash": key_hash_hex(tag.public_key)[:16], "battery": tag.battery,
                           "separated": tag.separated}
    if as_json:
        out.write(json.dumps(rec) + "\n")
        return
    line = (f"{rec['t_ms']:10.3f} ms  ch{rec['ch']}  {rec['pdu']:<15} {rec['addr'] or '-':<17}  "
            f"{rec['rssi_dbfs']:6.1f} dBFS  cfo {rec['cfo_khz']:+7.1f} kHz  crc {rec['crc']}")
    if tag is not None:
        m = rec["meshfind"]
        line += f"  meshfind H={m['key_hash']} batt={m['battery']}{' SEPARATED' if m['separated'] else ''}"
    out.write(line + "\n")


def cmd_rx_ble(a) -> int:
    from .ble.linklayer import channel_freq
    from .io import iq, sigmf
    from .io.sources import FileSource, HackRFSource, PlutoSource, stdin_source
    from .io.stream import StreamingBleReceiver

    center = a.center_freq if a.center_freq else channel_freq(a.channel)
    rate = a.rate
    if a.source == "hackrf":
        src = HackRFSource(center, rate, a.lna, a.vga, a.amp,
                           num_samples=int(a.seconds * rate) if a.seconds else None)
    elif a.source == "pluto":
        src = PlutoSource(center, rate, uri=a.uri, gain_db=a.gain)
    elif a.source == "file":
        if not a.file:
            print("--file is required with --source file", file=sys.stderr)
            return 2
        path = a.file
        fmt = a.format
        if path.endswith(".sigmf-meta") or path.endswith(".sigmf-data"):
            _, meta = sigmf.read(path)
            rate = meta["global"]["core:sample_rate"]
            center = meta["captures"][0]["core:frequency"]
            fmt = {"ci8": "cs8", "ci16_le": "cs16", "cf32_le": "cf32"}[meta["global"]["core:datatype"]]
            path = sigmf.base_path(path) + ".sigmf-data"
        src = FileSource(path, fmt or iq.fmt_from_path(path))
    else:
        src = stdin_source(a.format or "cs8")
    rx = StreamingBleReceiver(rate, center, a.channel)
    n = n_mf = 0
    t0 = time.perf_counter()
    try:
        for p in rx.run(src.blocks()):
            n += 1
            n_mf += _is_mf(p)
            _print_packet(p, a.json)
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    wall = time.perf_counter() - t0
    secs = rx.samples_in / rate
    print(f"# {n} packets ({n_mf} meshfind) in {secs:.3f} s of samples; "
          f"decoder busy {rx.busy_s:.2f} s ({secs / max(rx.busy_s, 1e-9):.2f}x real time), wall {wall:.2f} s; "
          f"ring overruns {rx.ring.overruns}", file=sys.stderr)
    return 0


def _is_mf(p) -> bool:
    from .meshfind_bridge import HAVE_MESHFIND, decode_meshfind
    return bool(HAVE_MESHFIND and decode_meshfind(p) is not None)


def cmd_tx_ble(a) -> int:
    from .ble.linklayer import channel_freq
    from .io import iq, sigmf
    from .meshfind_bridge import synth_capture
    x, truth = synth_capture(a.count, fs=a.rate, ebn0_db=None if a.no_noise else a.ebn0,
                             channel=a.channel, seed=a.seed)
    peak = float(np.max(np.abs(x)))
    x = x / peak * 0.7  # headroom for 8-bit quantisation
    fmt = a.format or iq.fmt_from_path(a.output)
    if a.output.endswith(".sigmf-data") or a.output.endswith(".sigmf-meta"):
        ann = [sigmf.annotation(t.sample_index, int(376e-6 * a.rate), "meshfind advert",
                                comment=t.public_key.hex()) for t in truth]
        sigmf.write(a.output, x, fmt, a.rate, channel_freq(a.channel),
                    description="sdrmesh synthetic meshfind adverts (not a real capture)", annotations=ann)
    else:
        iq.write(a.output, x, fmt)
    print(f"wrote {len(x)} samples ({len(x) / a.rate * 1e3:.1f} ms, {fmt}) with {len(truth)} meshfind "
          f"adverts + 1 other advert at {'no noise' if a.no_noise else f'Eb/N0 {a.ebn0} dB'} to {a.output}")
    return 0


def _lora_params(a):
    from .lora.params import LoRaParams
    return LoRaParams(sf=a.sf, bw=a.bw, cr=a.cr, has_crc=not a.no_crc, implicit_header=False,
                      preamble_len=a.preamble, sync_word=int(a.sync_word, 0), os=a.os)


def cmd_lora_tx(a) -> int:
    from .io import iq
    from .lora.modem import modulate
    p = _lora_params(a)
    payload = bytes.fromhex(a.hex) if a.hex else a.text.encode()
    x = 0.7 * modulate(payload, p)
    iq.write(a.output, x, a.format or iq.fmt_from_path(a.output, "cf32"))
    print(f"SF{p.sf} BW{p.bw / 1e3:g}k CR4/{4 + p.cr} {len(payload)} B -> {len(x)} samples at "
          f"{p.fs / 1e3:g} kS/s, time on air {p.airtime(len(payload)) * 1e3:.2f} ms -> {a.output}")
    return 0


def cmd_lora_rx(a) -> int:
    from .io import iq
    from .lora.modem import LoRaReceiver
    p = _lora_params(a)
    x = iq.read(a.file, a.format or iq.fmt_from_path(a.file, "cf32"))
    for f in LoRaReceiver(p).decode(x):
        txt = f.payload.decode("utf-8", "replace") if all(32 <= c < 127 for c in f.payload) else f.payload.hex()
        print(f"sample {f.start}: len {f.length} CR4/{4 + f.cr} crc {'ok' if f.crc_ok else 'BAD'} "
              f"cfo {f.cfo_hz:+.1f} Hz snr {f.snr_db:.1f} dB fec_fixes {f.fec_errors}: {txt}")
    return 0


def cmd_airtime(a) -> int:
    from .lora.params import LoRaParams
    p = LoRaParams(sf=a.sf, bw=a.bw, cr=a.cr, has_crc=not a.no_crc, preamble_len=a.preamble)
    print(f"SF{p.sf} BW{p.bw / 1e3:g}k CR4/{4 + p.cr} LDRO={'on' if p.low_data_rate else 'off'}: "
          f"{a.len} B -> {p.payload_symbols(a.len)} payload symbols, {p.airtime(a.len) * 1e3:.3f} ms on air, "
          f"PHY bit rate {p.bitrate:.1f} bit/s")
    return 0


def cmd_link_budget(a) -> int:
    from .link import budget_row
    from .lora.params import LoRaParams
    print("| SF | BW kHz | CR | B/pkt | airtime ms | reports/h @1% | goodput bit/s |")
    print("|---|---|---|---|---|---|---|")
    for bw in (125e3, 250e3, 500e3):
        for sf in range(7, 13):
            r = budget_row(LoRaParams(sf=sf, bw=bw, cr=a.cr), a.reports)
            print(f"| {sf} | {bw / 1e3:g} | {r['cr']} | {r['payload_B']} | {r['airtime_ms']:.1f} | "
                  f"{r['reports_per_hour']:.0f} | {r['goodput_bps']:.0f} |")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sdrmesh", description="Software-defined BLE and LoRa PHYs for meshfind")
    ap.add_argument("--version", action="version", version=f"sdrmesh {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    rx = sub.add_parser("rx", help="receive").add_subparsers(dest="phy", required=True)
    rb = rx.add_parser("ble", help="decode BLE advertising packets (meshfind tags)")
    rb.add_argument("--source", choices=["hackrf", "pluto", "file", "stdin"], default="file")
    rb.add_argument("--file", help="IQ file (.cs8 .cs16 .cf32 or SigMF)")
    rb.add_argument("--format", choices=["cs8", "cs16", "cf32"])
    rb.add_argument("--channel", type=int, default=37, choices=[37, 38, 39])
    rb.add_argument("--rate", type=float, default=4e6, help="sample rate (integer multiple of 4 MS/s for wideband)")
    rb.add_argument("--center-freq", type=float, help="tuning frequency (default: the channel centre)")
    rb.add_argument("--lna", type=int, default=32, help="HackRF LNA gain (dB, 0-40 step 8)")
    rb.add_argument("--vga", type=int, default=20, help="HackRF VGA gain (dB, 0-62 step 2)")
    rb.add_argument("--amp", action="store_true", help="HackRF RF amplifier on")
    rb.add_argument("--seconds", type=float, help="HackRF: stop after this many seconds")
    rb.add_argument("--uri", default="ip:pluto.local", help="PlutoSDR libiio URI")
    rb.add_argument("--gain", type=float, default=40, help="PlutoSDR RX gain (dB)")
    rb.add_argument("--json", action="store_true", help="one JSON object per packet")
    rb.set_defaults(fn=cmd_rx_ble)

    tx = sub.add_parser("tx", help="synthesise").add_subparsers(dest="phy", required=True)
    tb = tx.add_parser("ble", help="write a synthetic capture of meshfind adverts")
    tb.add_argument("--meshfind-demo", action="store_true", default=True)
    tb.add_argument("-o", "--output", required=True)
    tb.add_argument("--format", choices=["cs8", "cs16", "cf32"])
    tb.add_argument("--count", type=int, default=12)
    tb.add_argument("--rate", type=float, default=4e6)
    tb.add_argument("--channel", type=int, default=37, choices=[37, 38, 39])
    tb.add_argument("--ebn0", type=float, default=20.0)
    tb.add_argument("--seed", type=int, default=1)
    tb.add_argument("--no-noise", action="store_true", help="clean signal, e.g. for hackrf_transfer -t")
    tb.set_defaults(fn=cmd_tx_ble)

    lo = sub.add_parser("lora", help="LoRa modem").add_subparsers(dest="op", required=True)
    for name in ("tx", "rx"):
        q = lo.add_parser(name)
        q.add_argument("--sf", type=int, default=7)
        q.add_argument("--bw", type=float, default=125e3)
        q.add_argument("--cr", type=int, default=1, help="1..4 = 4/5..4/8")
        q.add_argument("--os", type=int, default=4, help="oversampling (fs = os * bw)")
        q.add_argument("--preamble", type=int, default=8)
        q.add_argument("--sync-word", default="0x12")
        q.add_argument("--no-crc", action="store_true")
        q.add_argument("--format", choices=["cs8", "cs16", "cf32"])
        if name == "tx":
            q.add_argument("--text", default="hello from sdrmesh")
            q.add_argument("--hex")
            q.add_argument("-o", "--output", required=True)
            q.set_defaults(fn=cmd_lora_tx)
        else:
            q.add_argument("--file", required=True)
            q.set_defaults(fn=cmd_lora_rx)

    at = sub.add_parser("airtime", help="LoRa time on air")
    at.add_argument("--sf", type=int, default=7)
    at.add_argument("--bw", type=float, default=125e3)
    at.add_argument("--cr", type=int, default=1)
    at.add_argument("--len", type=int, default=20)
    at.add_argument("--preamble", type=int, default=8)
    at.add_argument("--no-crc", action="store_true")
    at.set_defaults(fn=cmd_airtime)

    lb = sub.add_parser("link-budget", help="meshfind reports over LoRa: airtime and throughput")
    lb.add_argument("--cr", type=int, default=1)
    lb.add_argument("--reports", type=int, default=2, help="meshfind report frames per LoRa packet (1 or 2)")
    lb.set_defaults(fn=cmd_link_budget)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
