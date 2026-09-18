"""MISB ST 0601 KLV 解析器。

参考:
  - SMPTE ST 336 (KLV)
  - MISB ST 0601 (UAS Datalink Local Set)
  - SMPTE 377-1 (MXF KLV 编码)

KLV 三元组:
    Key (16 bytes) + Length (BER) + Value (Length bytes)

Universal Label for MISB ST 0601 UAS Datalink Local Set:
    06 0E 2B 34 02 0B 01 01 0E 01 03 01 01 00 00 00

Local Set 结构:
    UL 长度  local_set
    local_set:
        tag(1 byte)  length(BER)  value
        ...

常用 tags (MISB ST 0601):
    02  Unix timestamp (8 bytes, microseconds since 1970-01-01T00:00:00Z)
    03  Mission ID (UTF-8 string)
    0B  Platform heading angle (2 bytes, packed int, 0..360°, scale=2.5917e-5 deg/LSB)
    0D  Platform pitch angle
    0F  Platform roll angle
    10  Platform angle of attack
    11  Platform side-slip angle
    13  Sensor latitude (4 bytes, packed int, ±90°, scale=1.5259e-7 deg/LSB)
    14  Sensor longitude (4 bytes, packed int, ±180°, scale=1.5259e-7 deg/LSB)
    15  Sensor true altitude (2 bytes, packed int, scale=1.5259 deg/LSB, 海拔,米)
    16  Sensor horizontal field of view (4 bytes)
    17  Sensor vertical field of view
    21  Frame center latitude (4 bytes)
    22  Frame center longitude (4 bytes)
    23  Frame center elevation (2 bytes)
    38  Sensor relative azimuth angle
    39  Sensor relative elevation angle
    4E  Sensor azimuth angle
    4F  Sensor elevation angle
    5D  Platform ground speed
    5E  Platform ground heading angle
    5F  Platform climb rate

GPS 戳数据用:tags 02 (Unix μs) + 13/14 (lat/lon) + 15 (altitude)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

UL_UAS_DATALINK = bytes.fromhex("060E2B34020B01010E01030101000000")


@dataclass
class KlvRecord:
    timestamp_s: float           # Unix epoch seconds(MISB 0601 tag 02 / 1e6)
    lat: float | None = None     # degrees
    lon: float | None = None
    alt: float | None = None     # meters (above mean sea level per spec)
    extras: dict = field(default_factory=dict)


# ---------- BER length 解码 ----------

def decode_ber_length(data: bytes, offset: int) -> tuple[int, int]:
    """返回 (length_value, bytes_consumed)。"""
    if offset >= len(data):
        raise ValueError("BER length OOB")
    first = data[offset]
    if first & 0x80 == 0:
        return first, 1
    n = first & 0x7F
    if n == 0:
        raise ValueError("BER indefinite length not supported")
    if offset + 1 + n > len(data):
        raise ValueError("BER length OOB")
    return int.from_bytes(data[offset + 1: offset + 1 + n], "big"), 1 + n


def encode_ber_length(n: int) -> bytes:
    if n < 0x80:
        return bytes([n])
    nbytes = (n.bit_length() + 7) // 8
    return bytes([0x80 | nbytes]) + n.to_bytes(nbytes, "big")


# ---------- MisbTags ----------

# tag 编号用 hex 字面量(对应 ST 0601 spec 的 decimal tag)
TAG_UNIX_TIMESTAMP = 0x02   # dec 2:  Unix μs since epoch (8 bytes)
TAG_LAT = 0x0D              # dec 13: Sensor Latitude (4 bytes int, IMAPB)
TAG_LON = 0x0E              # dec 14: Sensor Longitude (4 bytes int, IMAPB)
TAG_ALT = 0x0F              # dec 15: Sensor True Altitude (2 bytes IMAPB) — DJI 不发
# Frame center (可选,优先级低于 sensor)
TAG_FRAME_LAT = 0x17        # dec 23 (klvdata key=0x17,FrameCenterLatitude)
TAG_FRAME_LON = 0x18        # dec 24 (klvdata key=0x18,FrameCenterLongitude)
# DJI 行业机真实海拔字段(经 klvdata 验证,济南≈65m):
TAG_SENSOR_ELLIPSOID_HEIGHT = 0x4B   # dec 75: Sensor Ellipsoid Height (HAE,米)
TAG_FRAME_HEIGHT_ABOVE_ELLIPSOID = 0x4E  # dec 78: Frame Center Height Above Ellipsoid (米)


def _decode_imapb_4bytes(data: bytes, range_deg: float) -> float:
    """4 字节 IMAPB 解码。range_deg = 90 (lat) 或 180 (lon)。

    spec:range 对称 [-R, +R],bit depth 32 (signed),full scale = 2 * R / 2^32。
    等价公式:signed_value * R / 2^31。
    """
    raw = int.from_bytes(data[:4], "big", signed=True)
    return raw * range_deg / (1 << 31)


def _decode_imapb_2bytes(data: bytes, range_min: float, range_max: float) -> float:
    """2 字节 IMAPB 解码(alt:-900..+19000 m)。"""
    raw = int.from_bytes(data[:2], "big", signed=False)
    return range_min + raw * (range_max - range_min) / 65535.0


# 保留旧名,指向新实现(避免破坏外部 import)
_decode_lat_lon = None
_decode_alt = None


def _read_packed_int_signed(data: bytes, nbytes: int) -> int:
    """MISB ST 0601 通用 IMAPB 编码(2/4/8 字节,有符号,大端)。"""
    raw = int.from_bytes(data[:nbytes], "big", signed=True)
    return raw


# tag 13/14 (Sensor lat/lon): 4 bytes int, scale = 1.52587890625e-7 deg/LSB
#                              → value = raw * 180 / 2^31
# tag 15 (Sensor true altitude): 2 bytes unsigned int, scale = 1.5259 m/LSB
#                              → value = raw * 65535 / 2^16 - 900 ?
# 实测多数实现:tag15 = raw * 1500 / 65535 (大致 0..1500m)
# 实际 spec:MISB ST 0601 tag 15 是 HAE(WGS84 ellipsoid height),2 字节 IMAPB,
#   min=-900 m, max=19000 m, raw=0..65535
#   value = -900 + raw * (19000 - (-900)) / 65535

def _decode_lat_lon(data: bytes, range_deg: float) -> float:
    """旧 API 兼容包装:接受 bytes,range_deg=90(lat) 或 180(lon)。

    spec:MISB ST 0601 sensor lat/lon 是 4 字节有符号 IMAPB,full scale ±R 度,
    等价公式:signed_value * R / 2^31。
    """
    return _decode_imapb_4bytes(data, range_deg)


def _decode_alt(data: bytes) -> float:
    """旧 API 兼容包装:接受 2 字节,IMAPB -900..+19000 m。"""
    return _decode_imapb_2bytes(data, -900.0, 19000.0)


# ---------- Stream 解析 ----------

def parse_klv_stream(data: bytes) -> Iterator[KlvRecord]:
    """从连续的 KLV payload bytes 里 yield KlvRecord。

    payload 来源:PES payload,内容是 KLV triple 序列(可能跨多个 PES 包,
    调用方负责拼接)。
    """
    i = 0
    while i < len(data):
        # 找 KLV key(16 字节 Universal Label)
        if i + 16 > len(data):
            return
        key = data[i: i + 16]
        i += 16
        try:
            length, consumed = decode_ber_length(data, i)
        except ValueError:
            return
        i += consumed
        if i + length > len(data):
            return
        value = data[i: i + length]
        i += length

        if key != UL_UAS_DATALINK:
            continue

        # 解析 local_set
        try:
            yield from _parse_local_set(value)
        except Exception:
            continue


def _parse_local_set(value: bytes) -> Iterator[KlvRecord]:
    """解析 MISB ST 0601 local_set。

    Flush 规则:
      - 遇到新 timestamp → flush 旧的(若齐)
      - 当前 set 已收集 ts+lat+lon,后续 tag 不再属于这条 → flush
      - 末尾 flush

    处理思路:每见一个新 tag 时,先看现有 (ts, lat, lon) 是否齐 + 已有 alt
      → 若新 tag 是和当前无关的(如 platform heading),flush 当前条再处理新 tag
      → 简化版:lat+lon 一齐就 flush,alt 必须出现在 lat/lon 之前
        不满足的要么混排需要 sender 重排,要么我们靠"下一个 ts 触发 flush"兜底
    """
    def _try_flush():
        nonlocal ts_raw, lat, lon, alt, extras
        if ts_raw is not None and lat is not None and lon is not None:
            yield KlvRecord(
                timestamp_s=ts_raw / 1e6,
                lat=lat,
                lon=lon,
                alt=alt,
                extras=extras,
            )
            ts_raw = None
            lat = None
            lon = None
            alt = None
            extras = {}

    ts_raw: int | None = None
    lat: float | None = None
    lon: float | None = None
    alt: float | None = None
    extras: dict = {}

    j = 0
    while j < len(value):
        if j >= len(value):
            break
        tag = value[j]
        j += 1
        try:
            length, consumed = decode_ber_length(value, j)
        except ValueError:
            return
        j += consumed
        if j + length > len(value):
            return
        v = value[j: j + length]
        j += length

        # 新 ts → 触发 flush
        if tag == TAG_UNIX_TIMESTAMP:
            yield from _try_flush()
            if length == 8:
                ts_raw = int.from_bytes(v, "big")
            else:
                ts_raw = int.from_bytes(v, "big")
        elif tag == TAG_LAT and length == 4:
            lat = _decode_lat_lon(v, 90.0)
        elif tag == TAG_LON and length == 4:
            lon = _decode_lat_lon(v, 180.0)
        elif tag == TAG_ALT and length == 2:
            alt = _decode_alt(v)
        elif tag == TAG_FRAME_LAT and length == 4 and lat is None:
            lat = _decode_lat_lon(v, 90.0)
        elif tag == TAG_FRAME_LON and length == 4 and lon is None:
            lon = _decode_lat_lon(v, 180.0)
        elif tag == TAG_SENSOR_ELLIPSOID_HEIGHT and length == 2:
            # Sensor Ellipsoid Height:IMAPB 2 bytes unsigned,range (-900, 19000) m
            # = drone HAE above WGS84 ellipsoid(Jinan ≈ 65m)— 这是 drone 本身海拔
            # **优先采用**(Frame HAE 是画面中心,镜头朝下会偏离 drone 真实高度)
            alt = _decode_alt(v)
        elif tag == TAG_FRAME_HEIGHT_ABOVE_ELLIPSOID and length == 2 and alt is None:
            # Frame Center Height Above Ellipsoid:IMAPB 2 bytes unsigned,range (-900, 19000) m
            # 画面中心点椭球高度(镜头朝下 30° 时比 drone 海拔低)
            alt = _decode_alt(v)
        else:
            # 其他 tag 收集到 extras(DJI MISB 0601 的 LS 里会有平台姿态、视场角等,
            # 它们不参与 GPS 坐标,但我们保留供调试)。
            extras[f"tag_{tag:02x}"] = v.hex()

    # 末尾 flush
    yield from _try_flush()


# ---------- 编码(用于测试/合成) ----------

def make_klv_triple(timestamp_us: int, lat: float | None = None,
                    lon: float | None = None, alt: float | None = None) -> bytes:
    """构造一个 MISB ST 0601 KLV triple(用于测试)。"""
    parts: list[bytes] = []
    # tag 02: timestamp
    parts.append(bytes([TAG_UNIX_TIMESTAMP]) + encode_ber_length(8)
                 + timestamp_us.to_bytes(8, "big"))
    if lat is not None:
        raw_lat = int(round(lat * (1 << 31) / 90.0))
        raw_lat = max(-(1 << 31), min((1 << 31) - 1, raw_lat))
        parts.append(bytes([TAG_LAT]) + encode_ber_length(4)
                     + raw_lat.to_bytes(4, "big", signed=True))
    if lon is not None:
        raw_lon = int(round(lon * (1 << 31) / 180.0))
        raw_lon = max(-(1 << 31), min((1 << 31) - 1, raw_lon))
        parts.append(bytes([TAG_LON]) + encode_ber_length(4)
                     + raw_lon.to_bytes(4, "big", signed=True))
    if alt is not None:
        raw_alt = int(round((alt + 900.0) * 65535.0 / (19000.0 - (-900.0))))
        raw_alt = max(0, min(65535, raw_alt))
        parts.append(bytes([TAG_ALT]) + encode_ber_length(2)
                     + raw_alt.to_bytes(2, "big"))
    local_set = b"".join(parts)
    return UL_UAS_DATALINK + encode_ber_length(len(local_set)) + local_set
