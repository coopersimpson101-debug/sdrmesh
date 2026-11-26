"""Regenerate every figure in the README from real code output.

    python docs/assets/make_figures.py            # needs matplotlib and meshfind

* hero_spectrogram.png - spectrogram (own STFT) of a synthesised capture: three
  meshfind BLE adverts on channel 37 and a LoRa SF7 frame carrying a meshfind
  report frame, both made by this repo's modulators;
* ber_snr.png          - Monte Carlo BER/SER/PER from docs/results/*.json
  (tools/run_montecarlo.py) against the analytic curves of sdrmesh.theory;
* rx_terminal.svg      - the real stdout of ``sdrmesh rx ble`` on a synthetic capture.
"""

from __future__ import annotations

import html
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sdrmesh import theory  # noqa: E402
from sdrmesh.dsp import spectrogram  # noqa: E402

ACCENT = "#FACC15"
BG = "#0d1117"
FG = "#e6edf3"
MUTED = "#8b949e"
GRID = "#30363d"
SF_COLORS = ["#FACC15", "#F59E0B", "#F97316", "#EF4444", "#EC4899", "#A78BFA"]


def _style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.8)


def hero_spectrogram(path):
    from meshfind import mesh, ratchet, report

    from sdrmesh.ble.gfsk import modulate_pdu
    from sdrmesh.ble.pdu import adv_nonconn
    from sdrmesh.channel import awgn, sigma2_for_snr
    from sdrmesh.dsp.filters import freq_shift
    from sdrmesh.link import pack_reports
    from sdrmesh.lora.modem import modulate
    from sdrmesh.lora.params import LoRaParams
    from sdrmesh.meshfind_bridge import tag_advert

    rng = np.random.default_rng(7)
    kr = ratchet.OwnerKeyring(master=bytes(range(32)), epoch0=0)
    # --- BLE: 3 adverts at 4 MS/s, different CFOs
    fs_b = 4e6
    parts = [np.zeros(400, dtype=complex)]
    for i, cfo in enumerate((-60e3, 40e3, 110e3)):
        addr, payload, _ = tag_advert(kr.master, 900.0 * i)
        x = modulate_pdu(adv_nonconn(addr, payload), 37, sps=4, guard_symbols=0)
        parts += [freq_shift(x, cfo, fs_b), np.zeros(1400, dtype=complex)]
    ble = awgn(np.concatenate(parts), sigma2_for_snr(25, fs_b, 1e6), rng)
    # --- LoRa: SF7 / 125 kHz frame carrying one meshfind report frame, os = 8 (1 MS/s)
    key = kr.key(0)
    sighting = report.Sighting(0, 7, 120, 340, 0, 100, -70, 1)
    env = report.seal(key.public, sighting, eph_secret=bytes(32))
    frame = mesh.Frame(mesh.T_REPORT, 10, 2, mesh.report_msg_id(env), env).encode()
    p = LoRaParams(sf=7, bw=125e3, cr=1, os=8)
    lr = pack_reports([frame], 1, 2)[0].encode()
    lora = modulate(lr[:40], p, pad_symbols=1)  # first 40 bytes keep the plot readable
    lora = awgn(lora, sigma2_for_snr(15, p.fs, p.bw), rng)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), dpi=110, gridspec_kw={"width_ratios": [1, 1.25]})
    fig.patch.set_facecolor(BG)
    t, f, s = spectrogram(ble, fs_b, nfft=128, hop=16)
    axs[0].pcolormesh(t * 1e3, f / 1e6, s.T, cmap="inferno", vmin=-55, vmax=-12, shading="auto", rasterized=True)
    axs[0].set(xlabel="time (ms)", ylabel="frequency offset (MHz)", ylim=(-1.2, 1.2),
               title="BLE LE 1M: 3 meshfind adverts, ch 37, GFSK BT 0.5 h 0.5")
    lora = lora[: int(26e-3 * p.fs)]  # first 26 ms: 8 up-chirps, sync word, 2.25 down-chirps, data chirps
    t, f, s = spectrogram(lora, p.fs, nfft=256, hop=32)
    axs[1].pcolormesh(t * 1e3, f / 1e3, s.T, cmap="inferno", vmin=-38, vmax=-2, shading="auto", rasterized=True)
    axs[1].set(xlabel="time (ms)", ylabel="frequency offset (kHz)", ylim=(-80, 80),
               title="LoRa SF7/125 kHz: preamble, sync 0x12, 2.25 down-chirps, data")
    for ax in axs:
        _style(ax)
        ax.grid(False)
    fig.text(0.5, 0.005, "synthesised by sdrmesh (no hardware); own STFT, Hann window", color=MUTED,
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def _load(name):
    with open(os.path.join(ROOT, "docs", "results", name + ".json"), encoding="utf-8") as f:
        return json.load(f)


def ber_figure(path):
    ble, lser, lper = _load("ble_ber"), _load("lora_ser"), _load("lora_per")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), dpi=110)
    fig.patch.set_facecolor(BG)
    # --- BLE BER
    ax = axs[0]
    e = np.linspace(0, 16, 200)
    ax.semilogy(e, theory.ber_msk_coherent(e), color=MUTED, ls=":", lw=1.2, label="coherent MSK bound")
    ax.semilogy(e, theory.ber_bfsk_noncoherent(e), color=FG, ls="--", lw=1.2, label="non-coherent BFSK")
    styles = {"awgn": ("o", ACCENT), "cfo+sto": ("s", "#F97316"), "multipath": ("^", "#EC4899")}
    labels = {"awgn": "AWGN, 4 MS/s", "cfo+sto": "AWGN + 120 kHz CFO + 0.37 sample + 40 ppm",
              "multipath": "3-tap multipath + 50 kHz CFO"}
    for imp, (mk, col) in styles.items():
        rows = [r for r in ble["results"] if r["impairment"] == imp and r["errors"] > 0 and r.get("fs", 4e6) == 4e6]
        ax.semilogy([r["ebn0_db"] for r in rows], [r["ber"] for r in rows], mk + "-", color=col, ms=4, lw=1.3,
                    label=labels[imp])
    rows = [r for r in ble["results"] if r.get("fs") == 2e6 and r["errors"] > 0]
    ax.semilogy([r["ebn0_db"] for r in rows], [r["ber"] for r in rows], "D--", color="#A78BFA", ms=3.5, lw=1.1,
                label="AWGN, 2 MS/s front end (streaming default)")
    ax.set(xlabel="Eb/N0 (dB)", ylabel="bit error rate", ylim=(1e-5, 0.6), xlim=(0, 16),
           title="BLE LE 1M receiver (Monte Carlo)")
    # --- LoRa uncoded SER
    ax = axs[1]
    for k, sf in enumerate(range(7, 13)):
        rows = [r for r in lser["results"] if r["sf"] == sf and r["symbol_errors"] > 0]
        snr = np.linspace(min(r["snr_db"] for r in lser["results"] if r["sf"] == sf) - 0.5,
                          max(r["snr_db"] for r in lser["results"] if r["sf"] == sf) + 0.5, 120)
        ax.semilogy(snr, theory.lora_ser(sf, snr), color=SF_COLORS[k], lw=1, alpha=0.7)
        ax.semilogy([r["snr_db"] for r in rows], [r["ser"] for r in rows], "o", color=SF_COLORS[k], ms=4,
                    label=f"SF{sf}")
    ax.set(xlabel="SNR in 125 kHz (dB)", ylabel="symbol error rate", ylim=(1e-4, 1.1),
           title="LoRa uncoded SER: dots = sim, lines = theory")
    # --- LoRa PER
    ax = axs[2]
    for k, sf in enumerate(range(7, 13)):
        for cr, ls in ((1, "-"), (4, "--")):
            rows = sorted([r for r in lper["results"] if r["sf"] == sf and r["cr"] == cr], key=lambda r: r["snr_db"])
            ax.semilogy([r["snr_db"] for r in rows], [max(r["per"], 1e-3) for r in rows], ls, marker="o", ms=3,
                        color=SF_COLORS[k], lw=1.2, label=f"SF{sf}" if cr == 1 else None)
    ax.plot([], [], "-", color=MUTED, label="CR 4/5")
    ax.plot([], [], "--", color=MUTED, label="CR 4/8")
    ax.set(xlabel="SNR in 125 kHz (dB)", ylabel="packet error rate (floored at 1e-3)", ylim=(8e-4, 1.1),
           title="LoRa PER, full receiver, 20 B, CFO + STO")
    for ax in axs:
        _style(ax)
        leg = ax.legend(fontsize=7.5, facecolor=BG, edgecolor=GRID, labelcolor=FG, loc="lower left")
        leg.get_frame().set_alpha(0.9)
    m = ble["meta"]
    fig.text(0.5, 0.005, f"Monte Carlo, {m['generated']}, {m['machine']}, Python {m['python']}, numpy {m['numpy']}"
             " - generated by tools/run_montecarlo.py", color=MUTED, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def terminal_svg(path):
    with tempfile.TemporaryDirectory() as d:
        cap = os.path.join(d, "demo.cs8")
        env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
        tx = subprocess.run([sys.executable, "-m", "sdrmesh", "tx", "ble", "-o", cap, "--count", "6", "--seed", "3",
                             "--ebn0", "14"], capture_output=True, text=True, check=True, env=env)
        rx = subprocess.run([sys.executable, "-m", "sdrmesh", "rx", "ble", "--source", "file", "--file", cap],
                            capture_output=True, text=True, check=True, env=env)
    tx_line = tx.stdout.strip().replace(cap, "demo.cs8")
    lines = ["$ sdrmesh tx ble -o demo.cs8 --count 6 --seed 3 --ebn0 14", tx_line,
             "$ sdrmesh rx ble --source file --file demo.cs8"]
    lines += rx.stdout.rstrip().splitlines()
    summary = rx.stderr.strip().splitlines()[-1]
    summary = summary.split("; decoder busy")[0]  # timing varies run to run; keep the counts
    lines.append(summary)
    _write_terminal(path, lines)


def _wrap(s, n):
    out = []
    while len(s) > n:
        out.append(s[:n])
        s = "  " + s[n:]
    out.append(s)
    return out


def _write_terminal(path, lines):
    wrapped = []
    for ln in lines:
        wrapped += _wrap(ln, 150)
    w, lh = 1280, 17
    h = 44 + lh * len(wrapped) + 14
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
           f'<rect width="{w}" height="{h}" rx="8" fill="{BG}" stroke="{GRID}"/>',
           '<circle cx="20" cy="16" r="5.5" fill="#ff5f56"/><circle cx="38" cy="16" r="5.5" fill="#ffbd2e"/>'
           '<circle cx="56" cy="16" r="5.5" fill="#27c93f"/>',
           f'<text x="{w / 2}" y="20" fill="{MUTED}" font-size="12" text-anchor="middle" '
           'font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace">sdrmesh - real output</text>']
    y = 46
    for ln in wrapped:
        col = ACCENT if ln.startswith("$") else (MUTED if ln.startswith("#") or ln.startswith("wrote") else FG)
        if "meshfind H=" in ln:
            a, b = ln.split("meshfind H=", 1)
            body = (f'<tspan fill="{FG}">{html.escape(a)}</tspan>'
                    f'<tspan fill="{ACCENT}">meshfind H={html.escape(b)}</tspan>')
        else:
            body = f'<tspan fill="{col}">{html.escape(ln)}</tspan>'
        out.append(f'<text x="14" y="{y}" font-size="12.5" xml:space="preserve" '
                   f'font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace">{body}</text>')
        y += lh
    out.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def main():
    which = sys.argv[1:] or ["hero", "ber", "terminal"]
    if "hero" in which:
        hero_spectrogram(os.path.join(HERE, "hero_spectrogram.png"))
    if "ber" in which:
        ber_figure(os.path.join(HERE, "ber_snr.png"))
    if "terminal" in which:
        terminal_svg(os.path.join(HERE, "rx_terminal.svg"))
    print("figures written to", HERE)


if __name__ == "__main__":
    main()
