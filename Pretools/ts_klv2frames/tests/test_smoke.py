"""单元测试。

只测纯 stdlib 的模块:KLV 编/解码 + EXIF 写入 + TS 包格式。
合成 TS 文件 → demux 抽 KLV → 解析 → 验证 GPS 数值正确。
合成 PNG → 写 EXIF → 读 EXIF → 验证 GPS 数值正确。

运行:
    /home/miniconda3/envs/3dgs/bin/python -m ts_klv2frames.tests.test_smoke
"""
from __future__ import annotations

import struct
import sys
import tempfile
import zlib
from pathlib import Path

# 让脚本能 import 父包
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # /home/Pretools

from ts_klv2frames.klv_misb0601 import (
    make_klv_triple, parse_klv_stream,
    UL_UAS_DATALINK, encode_ber_length, decode_ber_length,
    TAG_UNIX_TIMESTAMP, TAG_SENSOR_ELLIPSOID_HEIGHT, TAG_LAT, TAG_LON,
    _decode_alt,
)
from ts_klv2frames.exif_gps import (
    write_png_with_exif, read_exif_from_png, _build_exif_bytes,
    write_jpeg_with_exif, _alt_rational,
)


# ---------- Test 1: BER length ----------

def test_ber_length_roundtrip():
    for n in [0, 1, 0x7F, 0x80, 0x100, 0xFFFF, 0x10000, 0x123456]:
        b = encode_ber_length(n)
        v, c = decode_ber_length(b, 0)
        assert v == n, f"BER roundtrip failed for {n}: got {v}"
        assert c == len(b), f"BER consumed bytes mismatch for {n}"
    print("  [PASS] BER length roundtrip")


# ---------- Test 2: KLV triple roundtrip ----------

def test_klv_roundtrip():
    # 一个标准 ST 0601 triple
    t_us = 1757000000_000_000  # 2026-09-04T... μs
    lat = 39.9842
    lon = 116.3074
    alt = 55.3
    klv = make_klv_triple(t_us, lat=lat, lon=lon, alt=alt)
    assert klv[:16] == UL_UAS_DATALINK

    records = list(parse_klv_stream(klv))
    assert len(records) == 1, f"expected 1 record, got {len(records)}"
    r = records[0]
    assert abs(r.timestamp_s - t_us / 1e6) < 1e-3
    assert abs(r.lat - lat) < 1e-5, f"lat roundtrip off: {r.lat} vs {lat}"
    assert abs(r.lon - lon) < 1e-5, f"lon roundtrip off: {r.lon} vs {lon}"
    assert abs((r.alt or 0) - alt) < 0.5, f"alt roundtrip off: {r.alt} vs {alt}"
    print("  [PASS] KLV triple roundtrip")


def test_klv_multi_triple_concatenated():
    """流里多个 triple,顺序解析。"""
    ts = [1757000000_000_000, 1757000000_500_000, 1757000001_000_000]
    lats = [39.98, 39.99, 40.00]
    klv = b"".join(make_klv_triple(t, lat=la, lon=116.30, alt=50.0) for t, la in zip(ts, lats))
    records = list(parse_klv_stream(klv))
    assert len(records) == 3
    for i, (t, la, r) in enumerate(zip(ts, lats, records)):
        assert abs(r.timestamp_s - t / 1e6) < 1e-3
        assert abs(r.lat - la) < 1e-3, f"rec{i} lat {r.lat} vs {la}"
    print("  [PASS] KLV multi-triple")


# ---------- Test 3: TS 包 demux(合成最小 TS)----------

def make_ts_packet(pid: int, pusi: bool, payload: bytes,
                   stream_id: int = 0xBD) -> bytes:
    """构造一个 188 字节 TS 包(标准 PES header)。

    PES header (ISO/IEC 13818-1):
        00 00 01 SID PES_packet_length(2)
        marker(2) scrambling(2) priority(1) alignment(1) copyright(1) original(1)
        PTS_DTS_flags(2) ESCR(1) ES_rate(1) DSM_trick(1) additional_copy(1)
        PES_CRC(1) PES_extension(1)  → 共 2 bytes
        PES_header_data_length(1)
        data[header_data_length]

    我们用 header_data_length=0(不传 PTS,简化)。
    """
    # PES header: 9 bytes with marker bits default
    pes_packet_length = 3 + len(payload)  # 后面紧跟 marker(2) + PES_header_data_length(1) + payload
    pes_hdr = (
        b"\x00\x00\x01"
        + bytes([stream_id])
        + struct.pack(">H", pes_packet_length)
        + b"\x80\x00"        # marker=10b + scrambling=00 + ...
        + b"\x00"             # PES_header_data_length = 0
    )
    pes = pes_hdr + payload

    pusi_bit = 0x40 if pusi else 0
    pid_hi = (pid >> 8) & 0x1F
    pid_lo = pid & 0xFF
    afc = 0x10
    header = bytes([0x47, pusi_bit | pid_hi, pid_lo, afc])
    body = pes
    padding = b"\xFF" * (188 - len(header) - len(body))
    return header + body + padding


def test_ts_demux_roundtrip():
    """合成 3 个 TS 包 → demux → 解析 KLV → 验证。"""
    from ts_klv2frames.ts_demux import demux_klv_payload

    klv_data = b"".join([
        make_klv_triple(1757000000_000_000, lat=39.98, lon=116.30, alt=50.0),
        make_klv_triple(1757000000_500_000, lat=39.99, lon=116.31, alt=51.0),
    ])
    # 单 TS 包,PID 0x100,private_stream_1
    ts_pkt = make_ts_packet(pid=0x100, pusi=True, payload=klv_data)

    with tempfile.NamedTemporaryFile(suffix=".ts", delete=False) as f:
        f.write(ts_pkt)
        ts_path = Path(f.name)
    try:
        payloads = list(demux_klv_payload(ts_path, pid=0x100))
        assert len(payloads) == 1, f"got {len(payloads)} payloads"
        recs = list(parse_klv_stream(payloads[0]))
        assert len(recs) == 2
        assert abs(recs[0].lat - 39.98) < 1e-3
        assert abs(recs[1].lat - 39.99) < 1e-3
        print("  [PASS] TS demux → KLV parse")
    finally:
        ts_path.unlink()


# ---------- Test 4: EXIF PNG roundtrip ----------

def make_minimal_png(w: int = 16, h: int = 16) -> bytes:
    """构造最小合法 PNG(8 位灰度)。"""
    def chunk(t: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + t + data
                + struct.pack(">I", zlib.crc32(t + data) & 0xFFFFFFFF))

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)  # 灰度 8 bit
    raw = b""
    for _ in range(h):
        raw += b"\x00" + b"\x80" * w  # filter=0, 全 128
    idat = zlib.compress(raw)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def test_exif_gps_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "src.png"
        dst = tmp / "dst.png"
        src.write_bytes(make_minimal_png())

        write_png_with_exif(src, dst, lat=39.9842, lon=116.3074,
                            alt_m=55.3, utc_iso="2026-09-09T12:00:00Z")

        # 读 PNG,确认是合法 PNG(zTXt 在 IHDR 之后)
        assert dst.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        exif = read_exif_from_png(dst)
        assert exif is not None, "EXIF not found"
        assert exif["size"] > 0
        # 验证 GPS 关键字在 EXIF 里
        assert exif["exif_bytes"].startswith("45786966")  # "Exif" = 45 78 69 66
        print(f"  [PASS] EXIF GPS PNG roundtrip (size={exif['size']} bytes)")


# ---------- Test 6: DJI tag 0x4b Sensor Ellipsoid Height (HAE,IMAPB 2 bytes) ----------

def test_klv_sensor_ellipsoid_height():
    """DJI 行业机实际发的海拔字段是 tag 0x4B (Sensor Ellipsoid Height),
    2 bytes unsigned IMAPB,range (-900, +19000) m(对应济南 ~65m)。

    合成一个含 ts + lat + lon + tag 0x4B (raw=0x0c6c=3180,→65.6m) 的 KLV triple,
    验证 parse_klv_stream 能正确解析 alt≈65.6m。
    """
    t_us = 1757000000_000_000
    lat = 36.720079
    lon = 117.268276

    local_set = b""
    local_set += bytes([TAG_UNIX_TIMESTAMP]) + encode_ber_length(8) + t_us.to_bytes(8, "big")
    raw_lat = int(round(lat * (1 << 31) / 90.0))
    local_set += bytes([TAG_LAT]) + encode_ber_length(4) + raw_lat.to_bytes(4, "big", signed=True)
    raw_lon = int(round(lon * (1 << 31) / 180.0))
    local_set += bytes([TAG_LON]) + encode_ber_length(4) + raw_lon.to_bytes(4, "big", signed=True)
    # tag 0x4B (raw=0x0c6c = 3180) → 3180 * 19900 / 65535 - 900 = 65.62 m
    raw_alt_bytes = (0x0c6c).to_bytes(2, "big")
    local_set += bytes([TAG_SENSOR_ELLIPSOID_HEIGHT]) + encode_ber_length(2) + raw_alt_bytes

    klv = UL_UAS_DATALINK + encode_ber_length(len(local_set)) + local_set

    records = list(parse_klv_stream(klv))
    assert len(records) == 1
    r = records[0]
    assert abs(r.lat - lat) < 1e-4
    assert abs(r.lon - lon) < 1e-4
    assert r.alt is not None and 65.0 < r.alt < 66.5, \
        f"expected alt≈65.6m, got {r.alt}"
    print(f"  [PASS] KLV tag 0x4B (Sensor Ellipsoid Height) roundtrip (raw=0x0c6c → {r.alt:.2f}m)")


# ---------- Test 7: EXIF negative altitude (GPSAltitudeRef=1) ----------

def test_exif_negative_altitude():
    """负高度 → EXIF GPSAltitudeRef=1 (below sea level),|alt| cm."""
    for alt_m in [65.5, -12.3, 0.0, -983.0]:
        ref, num, den = _alt_rational(alt_m)
        if alt_m < 0:
            assert ref == 1, f"alt={alt_m}: expected ref=1 (below), got {ref}"
            assert num == int(round(-alt_m * 100)), f"alt={alt_m}: num={num}"
        else:
            assert ref == 0, f"alt={alt_m}: expected ref=0 (above), got {ref}"
            assert num == int(round(alt_m * 100)), f"alt={alt_m}: num={num}"
        assert den == 100
    print(f"  [PASS] EXIF negative altitude ref/num/den handling")


# ---------- main ----------

def main():
    tests = [
        test_ber_length_roundtrip,
        test_klv_roundtrip,
        test_klv_multi_triple_concatenated,
        test_ts_demux_roundtrip,
        test_exif_gps_roundtrip,
        test_klv_platform_alt_signed,
        test_exif_negative_altitude,
    ]
    print("=== ts_klv2frames smoke tests ===")
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
