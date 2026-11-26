"""Sample sources: files, stdin pipes, HackRF (``hackrf_transfer``) and
PlutoSDR (pyadi-iio, optional). Each yields blocks of complex samples."""

from __future__ import annotations

import shutil
import subprocess
import sys
from typing import Iterator

import numpy as np

from . import iq


class FileSource:
    def __init__(self, path: str, fmt: str | None = None, block: int = 1 << 18):
        self.path = path
        self.fmt = fmt or iq.fmt_from_path(path)
        self.block = block

    def blocks(self) -> Iterator[np.ndarray]:
        item = 2 * np.dtype(iq.FORMATS[self.fmt][1]).itemsize
        with open(self.path, "rb") as f:
            while True:
                raw = f.read(self.block * item)
                if not raw:
                    return
                yield iq.from_bytes(raw, self.fmt)


class PipeSource:
    """Raw interleaved samples from a binary stream (stdin, a subprocess)."""

    def __init__(self, stream, fmt: str = "cs8", block: int = 1 << 18):
        self.stream = stream
        self.fmt = fmt
        self.block = block

    def blocks(self) -> Iterator[np.ndarray]:
        item = 2 * np.dtype(iq.FORMATS[self.fmt][1]).itemsize
        want = self.block * item
        carry = b""
        while True:
            raw = self.stream.read(want)
            if not raw:
                break
            raw = carry + raw
            cut = (len(raw) // item) * item
            carry = raw[cut:]
            if cut:
                yield iq.from_bytes(raw[:cut], self.fmt)


def stdin_source(fmt: str = "cs8", block: int = 1 << 18) -> PipeSource:
    return PipeSource(sys.stdin.buffer, fmt, block)


def hackrf_command(freq: float, rate: float, lna_db: int = 32, vga_db: int = 20, amp: bool = False,
                   num_samples: int | None = None) -> list[str]:
    """``hackrf_transfer`` receive command streaming cs8 to stdout."""
    cmd = ["hackrf_transfer", "-r", "-", "-f", str(int(freq)), "-s", str(int(rate)),
           "-l", str(int(lna_db)), "-g", str(int(vga_db)), "-a", "1" if amp else "0"]
    if num_samples:
        cmd += ["-n", str(int(num_samples))]
    return cmd


class HackRFSource:
    """Live HackRF One via the ``hackrf_transfer`` CLI from libhackrf."""

    def __init__(self, freq: float, rate: float = 4e6, lna_db: int = 32, vga_db: int = 20,
                 amp: bool = False, block: int = 1 << 18, num_samples: int | None = None):
        self.cmd = hackrf_command(freq, rate, lna_db, vga_db, amp, num_samples)
        self.block = block
        self.proc = None

    def blocks(self) -> Iterator[np.ndarray]:
        if shutil.which("hackrf_transfer") is None:
            raise RuntimeError("hackrf_transfer not found on PATH (install the HackRF host tools)")
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            yield from PipeSource(self.proc.stdout, "cs8", self.block).blocks()
        finally:
            self.close()

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class PlutoSource:
    """ADALM-PLUTO via pyadi-iio (``pip install pyadi-iio``; needs libiio)."""

    def __init__(self, freq: float, rate: float = 4e6, uri: str = "ip:pluto.local", gain_db: float = 40,
                 block: int = 1 << 18, rf_bw: float | None = None):
        self.freq, self.rate, self.uri, self.gain, self.block = freq, rate, uri, gain_db, block
        self.rf_bw = rf_bw or rate

    def blocks(self) -> Iterator[np.ndarray]:
        try:
            import adi  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("PlutoSDR support needs pyadi-iio: pip install pyadi-iio") from exc
        sdr = adi.Pluto(self.uri)  # pragma: no cover - needs hardware
        sdr.sample_rate = int(self.rate)
        sdr.rx_lo = int(self.freq)
        sdr.rx_rf_bandwidth = int(self.rf_bw)
        sdr.rx_buffer_size = int(self.block)
        sdr.gain_control_mode_chan0 = "manual"
        sdr.rx_hardwaregain_chan0 = float(self.gain)
        while True:
            yield np.asarray(sdr.rx(), dtype=complex) / 2048.0  # 12-bit ADC
