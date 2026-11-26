"""Run the Monte Carlo campaigns behind the README plots and tables.

    python tools/run_montecarlo.py [--quick] [--jobs N]

Writes docs/results/{ble_ber,ble_per,lora_ser,lora_per}.json, each with the
machine, date, software versions and the exact settings. ``--quick`` runs a
smaller campaign (used by CI as a smoke test; results are not committed).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sdrmesh import montecarlo as mc  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "results")


def meta(extra: dict) -> dict:
    return {
        "generated": dt.date.today().isoformat(),
        "machine": f"{platform.processor() or platform.machine()} | {platform.system()} {platform.release()}",
        "python": platform.python_version(), "numpy": np.__version__,
        "tool": "tools/run_montecarlo.py", **extra,
    }


def _ble_ber(args):
    e, imp, quick, seed = args[:4]
    fs = args[4] if len(args) > 4 else 4e6
    target_err, max_frames = (40, 40) if quick else (200, 600)
    frames, errs, misses, bits = 0, 0.0, 0, 0
    while frames < max_frames:
        r = mc.ble_ber_point(e, 20, 1000, fs=fs, impairment=imp, seed=seed + frames)
        frames += 20
        errs += r["errors"]
        misses += r["sync_miss"]
        bits += r["bits"]
        if errs >= target_err:
            break
    return {"ebn0_db": e, "impairment": imp, "fs": fs, "bits": bits, "errors": errs, "ber": errs / bits, "sync_miss": misses}


def _ble_per(args):
    e, quick, seed = args
    from sdrmesh import channel as ch
    from sdrmesh.ble.gfsk import modulate_pdu
    from sdrmesh.ble.pdu import adv_nonconn
    from sdrmesh.ble.receiver import BleReceiver
    from sdrmesh.meshfind_bridge import decode_meshfind, tag_advert
    rng = np.random.default_rng(seed)
    n = 60 if quick else 400
    ok = 0
    rx = BleReceiver(4e6)
    for i in range(n):
        master = bytes(rng.integers(0, 256, 32, dtype=np.uint8))
        addr, payload, pub = tag_advert(master, 1000.0 * i)
        x = modulate_pdu(adv_nonconn(addr, payload), 37, sps=4, guard_symbols=0)
        y = ch.apply(x, 4e6, rng, cfo_hz=float(rng.uniform(-150e3, 150e3)), delay=float(rng.uniform(0, 4)),
                     lead=int(rng.integers(200, 2000)), tail=200, sigma2=ch.sigma2_for_ebn0(e, 4))
        pk = rx.decode(y)
        if pk and (t := decode_meshfind(pk[0])) is not None and t.public_key == pub:
            ok += 1
    return {"ebn0_db": e, "packets": n, "per": 1 - ok / n}


def _lora_ser(args):
    sf, s, quick, seed = args
    n = 3000 if quick else (40000 if sf <= 9 else 16000)
    return mc.lora_ser_point(sf, s, n, seed=seed)


def _lora_per(args):
    sf, cr, s, quick, seed = args
    n = 30 if quick else (300 if sf <= 9 else 100)
    return mc.lora_per_point(sf, cr, s, n, payload_len=20, os=2, cfo_bins_max=20.0, seed=seed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--only", nargs="*", default=["ble_ber", "ble_per", "lora_ser", "lora_per"])
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t_all = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        if "ble_ber" in a.only:
            t = time.time()
            grid = np.arange(0, 17, 1.0 if not a.quick else 4.0)
            jobs = [(float(e), imp, a.quick, 1000 * i + 17 * j) for j, imp in enumerate(mc.BLE_IMPAIRMENTS)
                    for i, e in enumerate(grid)]
            jobs += [(float(e), "awgn", a.quick, 5000 + 1000 * i, 2e6) for i, e in enumerate(grid)]
            res = list(ex.map(_ble_ber, jobs))
            _dump(a.out, "ble_ber", meta({"seconds": round(time.time() - t, 1), "fs": "4e6 unless the row says fs=2e6", "frame_bits": 1000,
                                          "impairments": {k: {kk: str(vv) for kk, vv in v.items()}
                                                          for k, v in mc.BLE_IMPAIRMENTS.items()}}), res)
        if "ble_per" in a.only:
            t = time.time()
            grid = np.arange(6, 17, 1.0 if not a.quick else 5.0)
            res = list(ex.map(_ble_per, [(float(e), a.quick, 77 + i) for i, e in enumerate(grid)]))
            _dump(a.out, "ble_per", meta({"seconds": round(time.time() - t, 1),
                                          "note": "meshfind adverts (37-byte PDU), CFO U(-150,150) kHz, "
                                                  "delay U(0,4) samples, random lead; success = CRC ok and "
                                                  "meshfind key recovered"}), res)
        if "lora_ser" in a.only:
            t = time.time()
            jobs = []
            for sf in range(7, 13):
                base = -6 - 2.5 * (sf - 7)
                for k, s in enumerate(np.arange(base - 10, base + 1.01, 1.0 if not a.quick else 5.0)):
                    jobs.append((sf, float(s), a.quick, 31 * sf + k))
            res = list(ex.map(_lora_ser, jobs))
            _dump(a.out, "lora_ser", meta({"seconds": round(time.time() - t, 1),
                                           "note": "uncoded, ideal sync, critically sampled, AWGN"}), res)
        if "lora_per" in a.only:
            t = time.time()
            jobs = []
            for sf in range(7, 13):
                base = -7.5 - 2.5 * (sf - 7)
                for cr in (1, 4):
                    for k, s in enumerate(np.arange(base - 6, base + 3.01, 1.0 if not a.quick else 4.5)):
                        jobs.append((sf, cr, float(s), a.quick, 1000 * sf + 100 * cr + k))
            res = list(ex.map(_lora_per, jobs))
            _dump(a.out, "lora_per", meta({"seconds": round(time.time() - t, 1),
                                           "note": "full receiver (detection, CFO/STO sync, header, FEC, CRC); "
                                                   "BW 125 kHz, os=2, 20-byte payload, CFO U(-20,20) bins, "
                                                   "fractional delay U(0,1) chip"}), res)
    print(f"total {time.time() - t_all:.0f} s")
    return 0


def _dump(d, name, m, rows):
    with open(os.path.join(d, name + ".json"), "w", encoding="utf-8") as f:
        json.dump({"meta": m, "results": rows}, f, indent=1)
        f.write("\n")
    print(f"{name}: {len(rows)} points in {m['seconds']} s")


if __name__ == "__main__":
    sys.exit(main())
