"""End to end with the real meshfind package: tag key -> BLE advert -> GFSK
-> channel -> our receiver -> meshfind decode; and relay report -> meshfind
mesh frame -> LR frame -> LoRa PHY -> channel -> receiver -> owner decrypts."""

import numpy as np
import pytest

from sdrmesh import channel as ch
from sdrmesh.ble.gfsk import modulate_pdu
from sdrmesh.ble.pdu import adv_nonconn
from sdrmesh.ble.receiver import BleReceiver
from sdrmesh.link import (BROADCAST, LRFrame, LinkError, MAX_LORA_PAYLOAD, T_ACK, T_DATA, budget_row, max_items,
                          pack_reports)
from sdrmesh.lora.modem import LoRaReceiver, modulate
from sdrmesh.lora.params import LoRaParams

mf = pytest.importorskip("meshfind")
from meshfind import adv, mesh, ratchet, report  # noqa: E402

from sdrmesh.meshfind_bridge import decode_meshfind, synth_capture, tag_advert  # noqa: E402


def _keyring():
    return ratchet.OwnerKeyring(master=bytes(range(32)), epoch0=1_700_000_000)


def test_tag_advert_built_by_meshfind_decodes_through_phy():
    kr = _keyring()
    rng = np.random.default_rng(0)
    for epoch in range(5):
        t = kr.epoch0 + epoch * ratchet.EPOCH_SECONDS + 3
        addr, payload, pub = tag_advert(kr.master, t, kr.epoch0, battery=2, separated=(epoch == 4))
        assert pub == kr.key(epoch if epoch < 4 else 0, separated=(epoch == 4)).public
        x = modulate_pdu(adv_nonconn(addr, payload), 39, sps=4, guard_symbols=0)
        y = ch.apply(x, 4e6, rng, cfo_hz=float(rng.uniform(-100e3, 100e3)), delay=float(rng.uniform(0, 4)),
                     lead=1000, tail=300, sigma2=ch.sigma2_for_ebn0(16, 4))
        pk = BleReceiver(4e6, channel=39).decode(y)
        assert len(pk) == 1 and pk[0].crc_ok
        tag = decode_meshfind(pk[0])
        assert tag is not None and tag.public_key == pub and tag.battery == 2 and tag.separated == (epoch == 4)


def test_synth_capture_all_tags_recovered():
    x, truth = synth_capture(20, ebn0_db=16, seed=4)
    pk = BleReceiver(4e6).decode(x)
    tags = [decode_meshfind(p) for p in pk]
    assert sum(t is not None for t in tags) == 20
    assert [t.public_key for t in tags if t] == [tr.public_key for tr in truth]
    assert any(t is None for t in tags)  # the non-meshfind advert was decoded but not mistaken for a tag


def test_lr_frame_codec():
    f = LRFrame(T_DATA, 7, BROADCAST, 200, [b"a" * 120, b"b" * 120])
    raw = f.encode()
    assert len(raw) == 250 <= MAX_LORA_PAYLOAD and LRFrame.decode(raw) == f
    a = LRFrame(T_ACK, 1, 7, 9, [200, 201])
    assert LRFrame.decode(a.encode()) == a
    assert max_items() == 2
    with pytest.raises(LinkError):
        LRFrame(T_DATA, 1, 2, 3, [b"x" * 120] * 3).encode()
    with pytest.raises(LinkError):
        LRFrame.decode(raw[:-1])
    with pytest.raises(LinkError):
        LRFrame.decode(b"\x00" + raw[1:])
    packed = pack_reports([b"r" * 120] * 5, 1, 2, seq0=10)
    assert [len(p.items) for p in packed] == [2, 2, 1] and [p.seq for p in packed] == [10, 11, 12]


def test_meshfind_report_over_lora_link_owner_decrypts():
    kr = _keyring()
    key = kr.key(3)
    sighting = report.Sighting(seen_ts=kr.epoch0 + 3 * 900 + 10, node_id=42, x=1234, y=-567, z=89,
                               accuracy_cm=150, rssi=-71, tag_status=adv.TagAdvert(key.public).status_bits)
    frames = []
    for _ in range(2):
        env = report.seal(key.public, sighting)
        frames.append(mesh.Frame(mesh.T_REPORT, mesh.DEFAULT_TTL, 2, mesh.report_msg_id(env), env).encode())
    lr = pack_reports(frames, src=0x0101, dst=0x0001)
    assert len(lr) == 1
    p = LoRaParams(sf=9, bw=125e3, cr=4, os=2)
    rng = np.random.default_rng(1)
    y = ch.apply(modulate(lr[0].encode(), p, pad_symbols=0), p.fs, rng, cfo_hz=7.3e3, delay=3.3,
                 lead=3000, tail=4 * p.n * p.os, sigma2=ch.sigma2_for_snr(-9.0, p.fs, p.bw))
    got = LoRaReceiver(p).decode(y)
    assert len(got) == 1 and got[0].crc_ok
    rx = LRFrame.decode(got[0].payload)
    assert rx.src == 0x0101 and len(rx.items) == 2
    for item in rx.items:
        f = mesh.Frame.decode(item)
        assert f.ftype == mesh.T_REPORT
        assert report.open_(key.private, key.public, f.payload) == sighting


def test_budget_row_numbers():
    r = budget_row(LoRaParams(sf=7, bw=125e3, cr=1), 2, snr_req_db=-6.0)
    assert r["payload_B"] == 250
    assert r["airtime_ms"] == pytest.approx(LoRaParams(sf=7).airtime(250) * 1e3)
    assert r["reports_per_hour"] == pytest.approx(0.01 * 3600 / (r["airtime_ms"] / 1e3) * 2)
    assert r["sensitivity_dbm"] == pytest.approx(-174 + 10 * np.log10(125e3) + 6 - 6)
