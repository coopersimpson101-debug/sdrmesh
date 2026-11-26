import io
import json
import subprocess
import sys

import numpy as np
import pytest

from sdrmesh.ble.linklayer import channel_freq
from sdrmesh.io import iq, sigmf
from sdrmesh.io.ringbuf import RingBuffer
from sdrmesh.io.sources import FileSource, PipeSource, PlutoSource, hackrf_command
from sdrmesh.io.stream import StreamChannelizer, StreamingBleReceiver
from sdrmesh.dsp.filters import channel_select

mf = pytest.importorskip("meshfind")
from sdrmesh.meshfind_bridge import decode_meshfind, synth_capture  # noqa: E402


@pytest.mark.parametrize("fmt", ["cs8", "cs16", "cf32"])
def test_iq_roundtrip(tmp_path, fmt):
    rng = np.random.default_rng(0)
    x = 0.5 * (rng.standard_normal(1000) + 1j * rng.standard_normal(1000)) / 3
    p = tmp_path / f"x.{fmt}"
    iq.write(str(p), x, fmt)
    y = iq.read(str(p), fmt)
    tol = {"cs8": 1 / 127, "cs16": 1e-4, "cf32": 1e-6}[fmt]
    assert np.max(np.abs(x - y)) <= tol
    assert iq.fmt_from_path(str(p)) == fmt
    assert len(iq.read(str(p), fmt, offset=100, count=50)) == 50


def test_cs8_is_hackrf_layout():
    raw = iq.to_bytes(np.array([1 + 0j, -1j, 0.5 + 0.5j]), "cs8")
    assert list(np.frombuffer(raw, dtype=np.int8)) == [127, 0, 0, -127, 64, 64]
    assert np.allclose(iq.from_bytes(raw + b"\x05", "cs8"), [1, -1j, (64 + 64j) / 127])  # partial sample dropped


def test_sigmf_roundtrip(tmp_path):
    x = np.exp(1j * np.arange(500) / 7)
    data, meta = sigmf.write(str(tmp_path / "rec"), x, "cf32", 4e6, 2402e6, description="test",
                             annotations=[sigmf.annotation(10, 20, "pkt", 2401e6, 2403e6)])
    m = json.load(open(meta))
    assert m["global"]["core:datatype"] == "cf32_le" and m["global"]["core:version"] == "1.0.0"
    assert m["captures"][0]["core:frequency"] == 2402e6
    assert m["annotations"][0]["core:freq_lower_edge"] == 2401e6
    y, m2 = sigmf.read(data)
    assert np.allclose(x, y, atol=1e-6) and m2 == m
    side = sigmf.write_meta_for(str(tmp_path / "cap.cs8"), "cs8", 8e6, 2426e6)
    assert json.load(open(side))["global"]["core:dataset"] == "cap.cs8"


def test_ring_buffer_wrap_and_overrun():
    rb = RingBuffer(10, dtype=np.int64)
    rb.write(np.arange(7))
    assert list(rb.peek(3)) == [0, 1, 2]
    rb.consume(5)
    rb.write(np.arange(7, 14))  # wraps
    assert list(rb.peek()) == list(range(5, 14)) and rb.read_index == 5
    rb.write(np.arange(14, 20))  # overrun: oldest unread samples are lost and counted
    assert rb.overruns == 5 and list(rb.peek()) == list(range(10, 20)) and len(rb) == 10
    rb.write(np.arange(20, 45))  # longer than capacity
    assert list(rb.peek()) == list(range(35, 45))


def test_stream_channelizer_equals_batch():
    rng = np.random.default_rng(4)
    x = rng.standard_normal(20000) + 1j * rng.standard_normal(20000)
    sc = StreamChannelizer(8e6, 1e6, 4e6)
    parts = [sc.process(x[i:i + 3333]) for i in range(0, len(x), 3333)]
    y = np.concatenate(parts)
    # same as a one-shot NCO + FIR + decimate (up to the causal filter delay)
    k = np.arange(len(x))
    ref = np.convolve(x * np.exp(-2j * np.pi * 1e6 / 8e6 * k), sc.h)[:len(x)][::2]
    assert np.allclose(y, ref[:len(y)])
    with pytest.raises(ValueError):
        StreamChannelizer(4e6, 3e6, 4e6)


def test_streaming_receiver_block_boundaries():
    x, truth = synth_capture(12, ebn0_db=18, seed=5)
    for blk in (3000, 17000, None):
        s = StreamingBleReceiver(4e6, channel_freq(37), block=blk)
        got = []
        for i in range(0, len(x), 4999):
            got += s.push(x[i:i + 4999])
        got += s.flush()
        mf_pk = [p for p in got if decode_meshfind(p) is not None]
        assert len(mf_pk) == 12 and len(got) == 13
        assert all(abs(p.extra["time_s"] * 4e6 - t.sample_index) <= 4 for p, t in zip(mf_pk, truth))


def test_streaming_large_source_chunks_do_not_overrun():
    x, truth = synth_capture(10, ebn0_db=18, seed=6)
    x = np.concatenate([x] * 12)  # ~0.13 s
    s = StreamingBleReceiver(4e6, channel_freq(37))
    got = [p for p in s.run([x[i:i + 400000] for i in range(0, len(x), 400000)]) if decode_meshfind(p)]
    assert len(got) == 120 and s.ring.overruns == 0


def test_wideband_capture_off_centre_channel():
    """8 MS/s capture tuned 1 MHz below channel 37: the channelizer must shift,
    filter and decimate before the 4 MS/s receiver."""
    x, truth = synth_capture(5, ebn0_db=20, seed=9)
    X = np.fft.fft(x)
    n = len(x)
    Y = np.zeros(2 * n, dtype=complex)
    Y[:n // 2], Y[-n // 2:] = X[:n // 2], X[-n // 2:]
    y = np.fft.ifft(Y) * 2
    y = y * np.exp(2j * np.pi * 1e6 / 8e6 * np.arange(len(y)))
    s = StreamingBleReceiver(8e6, channel_freq(37) - 1e6)
    got = [p for p in s.run([y]) if decode_meshfind(p)]
    assert {decode_meshfind(p).public_key for p in got} == {t.public_key for t in truth}


def test_file_and_pipe_sources(tmp_path):
    x = np.exp(1j * np.arange(10000) / 3) * 0.5
    p = tmp_path / "a.cs8"
    iq.write(str(p), x)
    a = np.concatenate(list(FileSource(str(p), block=777).blocks()))
    b = np.concatenate(list(PipeSource(io.BytesIO(p.read_bytes()), "cs8", block=1000).blocks()))
    assert np.allclose(a, b) and len(a) == 10000


def test_hackrf_command_line():
    cmd = hackrf_command(2402e6, 4e6, 24, 30, amp=True, num_samples=4000000)
    assert cmd[:3] == ["hackrf_transfer", "-r", "-"]
    assert "2402000000" in cmd and "4000000" in cmd and cmd[cmd.index("-a") + 1] == "1"


def test_pluto_source_without_pyadi():
    try:
        import adi  # noqa: F401
        pytest.skip("pyadi-iio installed; hardware test is manual")
    except ImportError:
        with pytest.raises(RuntimeError, match="pyadi-iio"):
            next(PlutoSource(2402e6).blocks())


def test_cli_rx_from_file_and_stdin(tmp_path):
    from sdrmesh.cli import main
    out = tmp_path / "cap.cs8"
    assert main(["tx", "ble", "-o", str(out), "--count", "6", "--seed", "3"]) == 0
    r = subprocess.run([sys.executable, "-m", "sdrmesh", "rx", "ble", "--source", "file", "--file", str(out), "--json"],
                       capture_output=True, text=True, check=True)
    recs = [json.loads(line) for line in r.stdout.splitlines()]
    assert sum("meshfind" in rec for rec in recs) == 6 and len(recs) == 7
    assert "6 meshfind" in r.stderr
    r2 = subprocess.run([sys.executable, "-m", "sdrmesh", "rx", "ble", "--source", "stdin"],
                        input=out.read_bytes(), capture_output=True, check=True)
    assert r2.stdout.decode().count("meshfind H=") == 6


def test_cli_sigmf_output_and_lora(tmp_path, capsys):
    from sdrmesh.cli import main
    assert main(["tx", "ble", "-o", str(tmp_path / "d.sigmf-data"), "--count", "3", "--no-noise"]) == 0
    assert main(["rx", "ble", "--file", str(tmp_path / "d.sigmf-meta")]) == 0
    assert capsys.readouterr().out.count("meshfind H=") == 3
    assert main(["lora", "tx", "--sf", "8", "--text", "ping", "-o", str(tmp_path / "f.cf32")]) == 0
    assert main(["lora", "rx", "--sf", "8", "--file", str(tmp_path / "f.cf32")]) == 0
    assert "crc ok" in capsys.readouterr().out
    assert main(["airtime", "--sf", "7", "--len", "13"]) == 0
    assert "46.336 ms" in capsys.readouterr().out
    assert main(["link-budget"]) == 0
    assert "| 12 | 125 |" in capsys.readouterr().out


def test_channel_select_matches_stream():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(4000) + 1j * rng.standard_normal(4000)
    assert len(channel_select(x, 8e6, 1e6, 4e6, 2e6)) == 2000
