"""PNG EXIF GPSInfo 写入器(纯 stdlib,不依赖 piexif)。

PNG 不像 JPEG 那样有原生 EXIF,需要在 PNG 文件里插入:
    - 一个 zTXt chunk(关键字 "Raw profile type exif")  -- Adobe XMP 风格
    - 或者一个 iTXt/tEXt "XML:com.adobe.xmp" -- 同样不通用
    - **JPEG EXIF via APP1 in PNG 不可行**

PNG 里携带 EXIF 的标准做法是:
    1. 把 EXIF TIFF 字节流(标准 EXIF,IFD0 + GPS IFD)
    2. 包装成 zTXt chunk: keyword="Raw profile type exif", compression=0
    3. 放到 PNG 文件 IHDR 后第一个 IDAT 前

主流 viewer/工具能识别这种 PNG-EXIF(Linux: exiftool, libgexiv2;
Windows: 不支持;Web: 大多数浏览器忽略)。

如果 viewer 不识别,我们还有一个 fallback:写 sidecar .xmp 文件。

PNG 文件结构:
    8-byte signature: \x89PNG\r\n\x1a\n
    chunks: [length(4) type(4) data CRC(4)]+
    chunk types: IHDR, IDAT, IEND, ...
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Optional

# PNG signature
PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _deg_to_dms_rational(deg: float) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """度 → ((d_num, d_den), (m_num, m_den), (s_num, s_den)) — EXIF GPS 格式。

    每个 rational 用 (numerator, denominator) 表示。
    """
    deg = abs(deg)
    d = int(deg)
    m_float = (deg - d) * 60
    m = int(m_float)
    s = round((m_float - m) * 60 * 1000)
    return ((d, 1), (m, 1), (s, 1000))


def _alt_rational(alt: float) -> tuple[int, int, int]:
    """返回 (alt_ref, num, den)。

    alt_ref: 0=above sea level, 1=below(EXIF GPSAltitudeRef byte)
    alt 作为 rational(|alt| cm 表示 → /100 转米精度)
    """
    if alt < 0:
        return (1, int(round(-alt * 100)), 100)
    return (0, int(round(alt * 100)), 100)


# TIFF/EXIF constants
TIFF_MAGIC_LE = b"II"  # little-endian
TAG_GPS_IFD = 0x8825
TAG_IFD0_EXIF_IFD = 0x8769

GPS_TAGS = {
    0x0000: ("GPSVersionID", "BYTE", 4),
    0x0001: ("GPSLatitudeRef", "ASCII", 2),
    0x0002: ("GPSLatitude", "RATIONAL", 3),
    0x0003: ("GPSLongitudeRef", "ASCII", 2),
    0x0004: ("GPSLongitude", "RATIONAL", 3),
    0x0005: ("GPSAltitudeRef", "BYTE", 1),
    0x0006: ("GPSAltitude", "RATIONAL", 1),
    0x0007: ("GPSTimeStamp", "RATIONAL", 3),
    0x001D: ("GPSDateStamp", "ASCII", 11),
}


def _build_exif_bytes(lat: float, lon: float, alt_m: float,
                      utc_iso: Optional[str] = None) -> bytes:
    """构造 EXIF TIFF 字节流(little-endian,IFD0 → ExifIFD → GPS IFD 链)。

    返回:
        "Exif\\x00\\x00" + tiff_header + ifd0 + (GPS IFD 链)
    """
    # 准备 GPS 数据
    lat_ref = b"N" if lat >= 0 else b"S"
    lon_ref = b"E" if lon >= 0 else b"W"
    lat_dms = _deg_to_dms_rational(lat)
    lon_dms = _deg_to_dms_rational(lon)
    alt_ref, alt_num, alt_den = _alt_rational(alt_m)
    alt_rat = (alt_num, alt_den)

    # UTC 时间戳 → GPSTimeStamp (hour/min/sec, rationals) + GPSDateStamp (YYYY:MM:DD)
    gps_time: Optional[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = None
    gps_date: Optional[bytes] = None
    if utc_iso:
        # 格式: 2026-09-09T12:00:00Z
        try:
            date_part, time_part = utc_iso.split("T")
            h, m, s = time_part.rstrip("Z").split(":")
            gps_time = ((int(h), 1), (int(m), 1), (int(int(s) * 1000), 1000))
            gps_date = date_part.encode("ascii")
        except Exception:
            pass

    # TIFF header: "II" + 0x002A + offset_to_IFD0 (8 bytes total)
    # 规划内存布局:
    #   [0..7]:   TIFF header
    #   [8..N]:   IFD0
    #   [N..]:    GPS IFD
    #   [..]:     value area

    ifd0_offset = 8
    ifd0_count = 1  # 仅 ExifIFD 指针
    ifd0_size = 2 + ifd0_count * 12 + 4  # count(2) + entries(12 each) + next_ifd(4)
    gps_ifd_offset = ifd0_offset + ifd0_size

    # GPS IFD entries
    gps_entries: list[tuple[int, int, int, int]] = []  # (tag, type, count, value_or_offset)
    gps_value_data: list[bytes] = []
    cursor = gps_ifd_offset + 2 + len(GPS_TAGS) * 12 + 4  # GPS IFD 头部之后开始堆 value

    def _add_gps(tag: int, type_id: int, count: int, payload: bytes) -> None:
        nonlocal cursor
        if len(payload) <= 4:
            value_field = payload.ljust(4, b"\x00")
        else:
            value_field = struct.pack("<I", cursor)
            gps_value_data.append(payload)
            cursor += len(payload)
        gps_entries.append((tag, type_id, count, struct.unpack("<I", value_field)[0]))

    _add_gps(0x0000, 1, 4, struct.pack("BBBB", 2, 3, 0, 0))
    _add_gps(0x0001, 2, 2, lat_ref + b"\x00")
    _add_gps(0x0002, 5, 3, b"".join(struct.pack("<II", n, d) for n, d in lat_dms))
    _add_gps(0x0003, 2, 2, lon_ref + b"\x00")
    _add_gps(0x0004, 5, 3, b"".join(struct.pack("<II", n, d) for n, d in lon_dms))
    _add_gps(0x0005, 1, 1, bytes([alt_ref]))
    _add_gps(0x0006, 5, 1, struct.pack("<II", alt_rat[0], alt_rat[1]))
    if gps_time is not None:
        _add_gps(0x0007, 5, 3, b"".join(struct.pack("<II", n, d) for n, d in gps_time))
    if gps_date is not None:
        _add_gps(0x001D, 2, len(gps_date), gps_date)

    # IFD0 只有一个 ExifIFD 指针(tag 0x8769 = GPS IFD offset 我们直接用)
    # 简化:用 tag 0x8825 (GPSIFD) 直接挂在 IFD0
    ifd0_entries = [(0x8825, 4, 1, gps_ifd_offset)]

    # 编码
    buf = bytearray()
    # TIFF header
    buf += TIFF_MAGIC_LE
    buf += struct.pack("<H", 0x002A)
    buf += struct.pack("<I", ifd0_offset)

    # IFD0
    buf += struct.pack("<H", len(ifd0_entries))
    for tag, typ, count, val in ifd0_entries:
        buf += struct.pack("<HHI", tag, typ, count) + struct.pack("<I", val)
    buf += struct.pack("<I", 0)  # next IFD = 0

    # GPS IFD
    buf += struct.pack("<H", len(gps_entries))
    for tag, typ, count, val in gps_entries:
        buf += struct.pack("<HHI", tag, typ, count) + struct.pack("<I", val)
    buf += struct.pack("<I", 0)  # next IFD = 0

    # value area
    buf += b"".join(gps_value_data)

    # EXIF 头:"Exif\0\0" + TIFF
    return b"Exif\x00\x00" + bytes(buf)


def _crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def write_png_with_exif(src_png: Path, dst_png: Path, lat: float, lon: float,
                        alt_m: float, utc_iso: Optional[str] = None) -> None:
    """在 src PNG 上嵌入 EXIF GPS 块,写入 dst。

    实现:
        读 src → 在 IHDR 之后插入一个 zTXt chunk("Raw profile type exif", exif_bytes)
              → 写到 dst
    """
    exif_bytes = _build_exif_bytes(lat, lon, alt_m, utc_iso)

    raw = src_png.read_bytes()
    if raw[:8] != PNG_SIG:
        raise ValueError(f"{src_png}: not a PNG")

    # zTXt chunk:
    #   keyword (latin1) + \x00 + compression_method(1) + compressed_text
    keyword = b"Raw profile type exif"
    compressed = zlib.compress(exif_bytes)
    ztxt_data = keyword + b"\x00" + b"\x00" + compressed
    chunk_type = b"zTXt"
    chunk_body = ztxt_data
    chunk = (struct.pack(">I", len(chunk_body)) + chunk_type + chunk_body
             + struct.pack(">I", _crc32(chunk_type + chunk_body)))

    # 找 IHDR 后位置插入(IHDR 一定在 8 字节后,是第一个 chunk)
    # IHDR chunk: 4(len) + 4(type) + 13(data) + 4(CRC) = 25 bytes
    # 8 (sig) + 25 (IHDR) = 33 → 在 offset 33 处插入 zTXt
    insert_at = 8 + 4 + 4 + 13 + 4
    out = raw[:insert_at] + chunk + raw[insert_at:]
    dst_png.write_bytes(out)


def read_exif_from_png(png_path: Path) -> dict | None:
    """读 PNG 里的 EXIF(debug 用)。"""
    raw = png_path.read_bytes()
    if raw[:8] != PNG_SIG:
        return None
    i = 8
    while i < len(raw):
        length = struct.unpack(">I", raw[i: i + 4])[0]
        ctype = raw[i + 4: i + 8]
        cdata = raw[i + 8: i + 8 + length]
        if ctype == b"zTXt":
            # keyword\0 compression_method compressed_text
            null_idx = cdata.find(b"\x00")
            if null_idx < 0:
                continue
            keyword = cdata[:null_idx]
            if keyword == b"Raw profile type exif":
                cm = cdata[null_idx + 1]
                compressed = cdata[null_idx + 2:]
                if cm == 0:
                    exif = zlib.decompress(compressed)
                    return {"exif_bytes": exif.hex(), "size": len(exif)}
        i += 8 + length + 4
    return None


# ============================================================
# JPEG EXIF(APP1 marker 0xFFE1)写入 —— 给 COLMAP / 所有 viewer 用
# ============================================================

JPEG_SOI = b"\xFF\xD8"
JPEG_EOI = b"\xFF\xD9"
APP1_MARKER = b"\xFF\xE1"


def write_jpeg_with_exif(jpeg_bytes: bytes, lat: float, lon: float,
                          alt_m: float, utc_iso: Optional[str] = None) -> bytes:
    """把 JPEG 字节流加上 EXIF APP1 段(含 GPSInfo)。

    标准 JPEG APP1 结构:
        FF E1 [length:2] "Exif\x00\x00" + TIFF header + IFD0 + GPS IFD

    length 字段包含 length 自身 2 字节,即:
        length = 2 + 6 (Exif\0\0) + TIFF body size
    """
    exif_payload = _build_exif_bytes(lat, lon, alt_m, utc_iso)
    # APP1 length = 2 (length itself) + len(exif_payload)
    length = 2 + len(exif_payload)
    app1 = APP1_MARKER + struct.pack(">H", length) + exif_payload

    # 在 SOI 后插入 APP1
    if not jpeg_bytes.startswith(JPEG_SOI):
        raise ValueError("not a JPEG (no SOI)")
    return jpeg_bytes[:2] + app1 + jpeg_bytes[2:]


def write_jpeg_with_gps_to_file(dst: Path, bgr_image, lat: float, lon: float,
                                 alt_m: float, utc_iso: Optional[str] = None,
                                 quality: int = 95) -> None:
    """cv2 写 JPEG + 注入 EXIF APP1 到文件。"""
    import cv2 as _cv2
    ok, buf = _cv2.imencode(".jpg", bgr_image,
                             [_cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise IOError(f"cv2.imencode jpg failed for {dst}")
    raw = buf.tobytes()
    out = write_jpeg_with_exif(raw, lat, lon, alt_m, utc_iso)
    dst.write_bytes(out)
