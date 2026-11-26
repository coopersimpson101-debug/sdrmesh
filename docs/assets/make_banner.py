"""Generate docs/assets/banner.svg (1280x400).

The motif on the right is real modulator output: the instantaneous frequency
(FM discriminator of the IQ) of a LoRa SF7 frame from ``sdrmesh.lora.modem``
(preamble up-chirps, sync word, down-chirps, data chirps) and, below it, the
GFSK frequency trace of a meshfind BLE advert from ``sdrmesh.ble.gfsk``.

    python docs/assets/make_banner.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from sdrmesh.ble.gfsk import modulate_pdu  # noqa: E402
from sdrmesh.ble.pdu import adv_nonconn  # noqa: E402
from sdrmesh.ble.receiver import discriminate  # noqa: E402
from sdrmesh.lora.modem import modulate  # noqa: E402
from sdrmesh.lora.params import LoRaParams  # noqa: E402

ACCENT = "#FACC15"
FONT = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"


def lora_trace():
    p = LoRaParams(sf=7, bw=125e3, cr=1, os=4)
    x = modulate(b"meshfind", p, pad_symbols=0)
    n_keep = int(19.5 * p.n * p.os)  # 8 up, 2 sync, 2.25 down, ~7 data chirps
    f = np.angle(x[1:n_keep] * np.conj(x[:n_keep - 1])) / (2 * math.pi) * p.os  # in units of BW, -0.5..0.5
    return f


def ble_trace():
    # a meshfind-shaped advert: static random address + 31-byte manufacturer data
    addr = bytes.fromhex("c6e2a91f3b70")
    payload = bytes([0x1E, 0xFF, 0xFF, 0xFF, 0x01]) + bytes(range(26))
    x = modulate_pdu(adv_nonconn(addr, payload), 37, sps=8, guard_symbols=0)
    d = discriminate(x, 8e6)
    return d[8 * 10: 8 * 10 + 8 * 120]  # 120 symbols from the preamble on


def polyline(xs, ys):
    return " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys))


def main():
    W, H = 1280, 400
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
           "<defs>",
           '<pattern id="g" width="32" height="32" patternUnits="userSpaceOnUse">'
           '<path d="M32 0H0V32" fill="none" stroke="#161b22" stroke-width="1"/></pattern>',
           '<linearGradient id="fade" x1="0" x2="1" y1="0" y2="0">'
           f'<stop offset="0" stop-color="{ACCENT}" stop-opacity="0.15"/>'
           f'<stop offset="0.25" stop-color="{ACCENT}" stop-opacity="1"/>'
           f'<stop offset="1" stop-color="{ACCENT}" stop-opacity="1"/></linearGradient>',
           "</defs>",
           f'<rect width="{W}" height="{H}" fill="#0d1117"/><rect width="{W}" height="{H}" fill="url(#g)"/>',
           f'<text x="72" y="182" font-family="{FONT}" font-weight="bold" font-size="104" fill="#e6edf3">sdrmesh</text>',
           f'<text x="76" y="234" font-family="{FONT}" font-size="22" fill="#8b949e">BLE and LoRa physical layers,</text>',
           f'<text x="76" y="268" font-family="{FONT}" font-size="22" fill="#8b949e">written from scratch, for the mesh</text>',
           f'<rect x="76" y="298" width="420" height="4" rx="2" fill="{ACCENT}"/>',
           f'<text x="76" y="338" font-family="{FONT}" font-size="16" fill="#8b949e">'
           "GFSK · CSS · sync · FEC · HackRF / PlutoSDR</text>"]
    # panel
    x0, x1 = 640, 1240
    out.append(f'<rect x="{x0 - 16}" y="36" width="{x1 - x0 + 32}" height="328" rx="8" fill="none" '
               'stroke="#30363d" stroke-dasharray="4 6"/>')
    f = lora_trace()
    xs = np.linspace(x0, x1, len(f))
    keep = np.arange(0, len(f), 3)
    yc, amp = 150, 92
    ys = yc - f * 2 * amp * 0.5 * 1.0
    # break the polyline at chirp wraps so the jump is not drawn
    seg = [keep[0]]
    pieces = []
    for a, b in zip(keep[:-1], keep[1:]):
        if abs(f[b] - f[a]) > 0.5:
            pieces.append(seg)
            seg = []
        seg.append(b)
    pieces.append(seg)
    for pc in pieces:
        if len(pc) > 1:
            out.append(f'<polyline points="{polyline(xs[pc], ys[pc])}" fill="none" stroke="url(#fade)" '
                       'stroke-width="2.2" stroke-linejoin="round"/>')
    out.append(f'<text x="{x0}" y="62" font-family="{FONT}" font-size="13" fill="#8b949e">'
               "LoRa SF7: f(t) of a real modulated frame</text>")
    d = ble_trace()
    xs = np.linspace(x0, x1, len(d))
    keep = np.arange(0, len(d), 2)
    ys = 300 - d[keep] * 26
    out.append(f'<polyline points="{polyline(xs[keep], ys)}" fill="none" stroke="{ACCENT}" stroke-opacity="0.75" '
               'stroke-width="1.6" stroke-linejoin="round"/>')
    out.append(f'<text x="{x0}" y="352" font-family="{FONT}" font-size="13" fill="#8b949e">'
               "BLE GFSK: f(t) of a meshfind advert, +-250 kHz</text>")
    out.append("</svg>")
    path = os.path.join(HERE, "banner.svg")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
