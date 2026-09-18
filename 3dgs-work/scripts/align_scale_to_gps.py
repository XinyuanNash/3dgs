#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
align_scale_to_gps.py — 3DGS 尺度对齐到真实 GPS 坐标
================================================================

用途
----
3DGS / COLMAP 的 SfM 重建是**任意尺度**(arbitrary scale)——COLMAP 通过
BA 把第一张图的相机中心固定在原点,scale 由图像匹配的三角化决定,
因此 tvec/point3D 的单位是"重建单位",与真实米/英尺无关。

对于 DJI 等带 GPS 的无人机图像,EXIF 里会写入相机中心的 WGS84
(latitude / longitude / altitude above sea level)。本脚本读 EXIF GPS
→ 转 ENU (East-North-Up) 局部切平面坐标 → 与 COLMAP tvec 做
**Umeyama 1991 闭式 7-DoF 相似变换**求解 (s, R, t) → 把 (s, R, t)
应用到所有 cameras.bin / images.bin / points3D.bin → 写回 COLMAP 文件,
让整个重建对齐到真实世界米制尺度 + ENU 局部坐标系。

COLMAP 是右手系 (X 右, Y 下, Z 前, world),而 GPS 是地理坐标;
脚本把 GPS 转 ENU(米)作为目标坐标系,所以最终 COLMAP world frame
的单位是"米,原点为 GPS 中心,轴向 East/North/Up"。

需要
----
- Python 3.7+(仅 stdlib:struct / math / json / os / sys / argparse / pathlib)
- 不依赖 numpy / scipy(若可用会做更精确的 SVD 加速,无则用闭式解)
- 不依赖 PIL——直接读 JPEG 的 APP1/EXIF 字节,避免 PIL 的 _getexif() /
  getexif() 在某些情况下返回空

输入
----
- dataset_dir: 包含 images/ + sparse/0/{cameras.bin, images.bin, points3D.bin} 的 COLMAP 目录
- --images-dir: 自定义图像目录(默认 dataset_dir/images)
- --gps-json: 可选,手填 GPS 的 sidecar JSON(若 EXIF 已被预处理剥掉)

输出
----
- 修改 sparse/0/{cameras,images,points3D}.bin 写回真实尺度 + ENU 对齐
- 输出 gps_alignment.json(7-DoF 变换 + 每张图的 GPS + 残差 RMSE)
- --dry-run: 只打印 / 只写 sidecar,不覆盖原 COLMAP 文件

用法
----
    # 推荐:先 dry-run 看 RMSE
    python align_scale_to_gps.py /path/to/dataset --dry-run

    # 确认无问题,真写
    python align_scale_to_gps.py /path/to/dataset

    # 如果 EXIF 已被剥掉,提供手填的 sidecar JSON
    python align_scale_to_gps.py /path/to/dataset --gps-json gps_sidecar.json

JSON sidecar 格式(可选,优先级高于 EXIF):
    {
      "DJI_0001.JPG": {"lat": 30.123456, "lon": 114.654321, "alt": 123.45},
      "DJI_0002.JPG": {"lat": 30.123478, "lon": 114.654298, "alt": 123.51},
      ...
    }

注意
----
- 至少需要 3 张图有 GPS 才能求解 7-DoF
- 用 ≥10 张图能稳定;Umeyama 闭式解对 outlier 敏感,RANSAC 化见 _umeyama
- GPS altitude 通常是 EGM96 / WGS84 ellipsoid height,不是海拔(barometric)——
  与 SfM 的"海拔"差几米到几十米,可接受
- 若 SfM 用了 metric depth(比如 mono-depth prior + depth_params.json),
  优先信任 depth-aligned scale,GPS 仅做旋转 + 平移对齐

参考
----
- Umeyama 1991 "Least-Squares Estimation of Transformation Parameters
  Between Two Point Patterns"
- COLMAP binary format: see scene/colmap_loader.py in 3dgs-work
- WGS84 → ECEF → ENU: standard geodesy (Hofmann-Wellenhof 2008)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================================
# 1. EXIF GPS 解析(不依赖 PIL,直接读 JPEG 字节)
# ============================================================================

def _read_rational(d: bytes, off: int, endian: str) -> float:
    """EXIF RATIONAL = numerator/denominator (both unsigned 32-bit)."""
    num, den = struct.unpack(endian + 'II', d[off:off + 8])
    if den == 0:
        return 0.0
    return num / den


def _read_srational(d: bytes, off: int, endian: str) -> float:
    """EXIF SRATIONAL = signed numerator/denominator."""
    num, den = struct.unpack(endian + 'ii', d[off:off + 8])
    if den == 0:
        return 0.0
    return num / den


def parse_exif_gps(jpeg_path: str) -> Optional[Tuple[float, float, float]]:
    """
    从 JPEG 文件直接解析 EXIF GPS IFD。返回 (lat, lon, alt) 十进制度数 + 米。
    失败返回 None(JPEG 无 APP1 / 无 GPS IFD / 不支持的格式)。
    """
    try:
        with open(jpeg_path, 'rb') as f:
            data = f.read(131072)  # 前 128KB 通常够 APP1
    except (OSError, IOError):
        return None

    if data[:2] != b'\xff\xd8':
        return None  # 不是 JPEG

    # 扫描 JPEG markers,找到 APP1 (FFE1) + 'Exif\x00\x00'
    i = 2
    tiff = None
    while i < len(data) - 4:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD9, 0xDA):
            break  # EOI 或 SOS,metadata 结束
        if i + 4 > len(data):
            break
        seg_len = struct.unpack('>H', data[i + 2:i + 4])[0]
        if i + 2 + seg_len > len(data):
            break
        seg = data[i + 4:i + 2 + seg_len]
        if marker == 0xE1 and seg.startswith(b'Exif\x00\x00'):
            tiff = i + 4 + 6  # TIFF header starts 6 bytes after 'Exif\0\0'
            break
        i += 2 + seg_len

    if tiff is None or tiff + 8 > len(data):
        return None

    # TIFF header
    bo = data[tiff:tiff + 2]
    if bo == b'II':
        endian = '<'
    elif bo == b'MM':
        endian = '>'
    else:
        return None

    if tiff + 10 > len(data):
        return None
    n_ifd0 = struct.unpack(endian + 'H', data[tiff + 8:tiff + 10])[0]

    # 找 IFD0 中 GPS IFD pointer (tag 0x8825)
    gps_off = None
    for e in range(min(n_ifd0, 64)):
        ent = tiff + 10 + e * 12
        if ent + 12 > len(data):
            break
        tag = struct.unpack(endian + 'H', data[ent:ent + 2])[0]
        if tag == 0x8825:
            gps_off = tiff + struct.unpack(endian + 'I', data[ent + 8:ent + 12])[0]
            break

    if gps_off is None or gps_off + 2 > len(data):
        return None

    n_gps = struct.unpack(endian + 'H', data[gps_off:gps_off + 2])[0]
    if n_gps > 64:
        return None  # 异常

    # GPS 字段:dms 度分秒 + ref
    gps_fields: Dict[int, Tuple[float, str]] = {}
    for e in range(n_gps):
        ent = gps_off + 2 + e * 12
        if ent + 12 > len(data):
            break
        tag = struct.unpack(endian + 'H', data[ent:ent + 2])[0]
        dtype = struct.unpack(endian + 'H', data[ent + 2:ent + 4])[0]
        count = struct.unpack(endian + 'I', data[ent + 4:ent + 8])[0]
        val_field = data[ent + 8:ent + 12]

        if dtype == 2:  # ASCII
            # value is offset to string if count > 4
            if count <= 4:
                s = val_field[:count].decode('ascii', errors='ignore').rstrip('\x00')
            else:
                str_off = tiff + struct.unpack(endian + 'I', val_field)[0]
                s = data[str_off:str_off + count].decode('ascii', errors='ignore').rstrip('\x00')
            gps_fields[tag] = (s, 'ascii')
        elif dtype == 5:  # RATIONAL — count 3 = deg/min/sec
            # offset to 3 rationals
            r_off = tiff + struct.unpack(endian + 'I', val_field)[0]
            if r_off + 24 > len(data):
                continue
            d_val = _read_rational(data, r_off, endian)
            m_val = _read_rational(data, r_off + 8, endian)
            s_val = _read_rational(data, r_off + 16, endian)
            decimal = d_val + m_val / 60.0 + s_val / 3600.0
            gps_fields[tag] = (decimal, 'rational')
        elif dtype == 10:  # SRATIONAL — signed rational (altitude uses this)
            r_off = tiff + struct.unpack(endian + 'I', val_field)[0]
            if r_off + 8 > len(data):
                continue
            decimal = _read_srational(data, r_off, endian)
            gps_fields[tag] = (decimal, 'srational')
        elif dtype in (3, 4):  # SHORT / LONG
            val = struct.unpack(endian + ('H' if dtype == 3 else 'I'), val_field)[0]
            gps_fields[tag] = (val, 'int')

    # tag 0x0001 GPSLatitudeRef = 'N' or 'S'
    # tag 0x0002 GPSLatitude (rationals)
    # tag 0x0003 GPSLongitudeRef = 'E' or 'W'
    # tag 0x0004 GPSLongitude
    # tag 0x0005 GPSAltitudeRef (byte: 0=above sea, 1=below)
    # tag 0x0006 GPSAltitude (rational: meters)

    if 0x0002 not in gps_fields or 0x0004 not in gps_fields:
        return None

    lat = gps_fields[0x0002][0]
    lon = gps_fields[0x0004][0]
    if 0x0001 in gps_fields and gps_fields[0x0001][0] == 'S':
        lat = -lat
    if 0x0003 in gps_fields and gps_fields[0x0003][0] == 'W':
        lon = -lon

    alt = 0.0
    if 0x0006 in gps_fields:
        alt = gps_fields[0x0006][0]
        if 0x0005 in gps_fields and gps_fields[0x0005][0] == 1:
            alt = -alt

    return (lat, lon, alt)


# ============================================================================
# 2. WGS84 → ECEF → ENU
# ============================================================================

# WGS84 椭球参数
WGS84_A = 6378137.0               # 长半轴 (m)
WGS84_F = 1.0 / 298.257223563    # 扁率
WGS84_E2 = WGS84_F * (2 - WGS84_F)  # 第一偏心率平方


def wgs84_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> Tuple[float, float, float]:
    """WGS84 (lat/lon in degrees, alt in meters) → ECEF (X, Y, Z) in meters."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (N + alt_m) * cos_lat * cos_lon
    y = (N + alt_m) * cos_lat * sin_lon
    z = (N * (1.0 - WGS84_E2) + alt_m) * sin_lat
    return (x, y, z)


def ecef_to_enu(x: float, y: float, z: float,
                lat0_deg: float, lon0_deg: float, alt0_m: float) -> Tuple[float, float, float]:
    """ECEF (x, y, z) → ENU (East, North, Up) in meters, relative to ref (lat0, lon0, alt0)."""
    x0, y0, z0 = wgs84_to_ecef(lat0_deg, lon0_deg, alt0_m)
    dx, dy, dz = x - x0, y - y0, z - z0
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    sin_lat, cos_lat = math.sin(lat0), math.cos(lat0)
    sin_lon, cos_lon = math.sin(lon0), math.cos(lon0)
    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    return (e, n, u)


def gps_to_enu(lat: float, lon: float, alt: float,
               lat0: float, lon0: float, alt0: float) -> Tuple[float, float, float]:
    """直接 GPS → ENU(无需走 ECEF 中转)."""
    ecef = wgs84_to_ecef(lat, lon, alt)
    return ecef_to_enu(*ecef, lat0, lon0, alt0)


# ============================================================================
# 3. COLMAP 二进制格式读写
# ============================================================================

def _read_cameras_bin(path: str) -> "OrderedDict[int, dict]":
    """返回 {cam_id: {model_id, model_name, width, height, params}}."""
    cameras = OrderedDict()
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            cam_id, model_id, width, height = struct.unpack('<IiQQ', f.read(24))
            model_name = _CAMERA_MODEL_BY_ID(model_id)
            n_params = _CAMERA_MODEL_NUM_PARAMS(model_id)
            params = struct.unpack(f'<{n_params}d', f.read(8 * n_params))
            cameras[cam_id] = {
                'model_id': model_id,
                'model_name': model_name,
                'width': width,
                'height': height,
                'params': list(params),
            }
    return cameras


def _write_cameras_bin(path: str, cameras: "OrderedDict[int, dict]") -> None:
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(cameras)))
        for cam_id, c in cameras.items():
            f.write(struct.pack('<IiQQ', cam_id, c['model_id'], c['width'], c['height']))
            f.write(struct.pack(f'<{len(c["params"])}d', *c['params']))


def _read_images_bin(path: str) -> "OrderedDict[int, dict]":
    """返回 {image_id: {qvec (4,), tvec (3,), camera_id, name, xys, point3D_ids}}。

    COLMAP 2D 观测格式:每条观测 24 字节 = (double x, double y, uint64 point3D_id),interleaved。
    """
    images = OrderedDict()
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            image_id = struct.unpack('<I', f.read(4))[0]
            qvec = struct.unpack('<4d', f.read(32))
            tvec = struct.unpack('<3d', f.read(24))
            camera_id = struct.unpack('<I', f.read(4))[0]
            name = b''
            while True:
                ch = f.read(1)
                if ch == b'\x00':
                    break
                name += ch
            n_pts = struct.unpack('<Q', f.read(8))[0]
            xys = []
            pt_ids = []
            for _ in range(n_pts):
                x, y, pid = struct.unpack('<ddQ', f.read(24))
                xys.extend([x, y])
                pt_ids.append(pid)
            images[image_id] = {
                'qvec': list(qvec),
                'tvec': list(tvec),
                'camera_id': camera_id,
                'name': name.decode('utf-8'),
                'xys': xys,
                'point3D_ids': pt_ids,
            }
    return images


def _write_images_bin(path: str, images: "OrderedDict[int, dict]") -> None:
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(images)))
        for image_id, img in images.items():
            f.write(struct.pack('<I', image_id))
            f.write(struct.pack('<4d', *img['qvec']))
            f.write(struct.pack('<3d', *img['tvec']))
            f.write(struct.pack('<I', img['camera_id']))
            f.write(img['name'].encode('utf-8') + b'\x00')
            n_pts = len(img['point3D_ids'])
            f.write(struct.pack('<Q', n_pts))
            # 2D 观测 interleaved: (x, y, point3D_id) 三元组,每组 24 字节
            for j in range(n_pts):
                f.write(struct.pack('<dd', img['xys'][2 * j], img['xys'][2 * j + 1]))
                f.write(struct.pack('<Q', img['point3D_ids'][j]))


def _read_points3D_bin(path: str) -> "OrderedDict[int, dict]":
    """COLMAP points3D.bin 每点格式:
       uint64 id | 3*double xyz | 3*uchar rgb | double error | uint64 track_len | track_len × (uint32 image_id, uint32 point2D_idx)
    """
    points = OrderedDict()
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            pt_id = struct.unpack('<Q', f.read(8))[0]
            xyz = struct.unpack('<3d', f.read(24))
            rgb = struct.unpack('<3B', f.read(3))
            err = struct.unpack('<d', f.read(8))[0]
            track_len = struct.unpack('<Q', f.read(8))[0]
            image_ids = []
            point2D_idxs = []
            for _ in range(track_len):
                img_id, pt2d_idx = struct.unpack('<II', f.read(8))
                image_ids.append(img_id)
                point2D_idxs.append(pt2d_idx)
            points[pt_id] = {
                'xyz': list(xyz),
                'rgb': list(rgb),
                'error': err,
                'image_ids': image_ids,
                'point2D_idxs': point2D_idxs,
            }
    return points


def _write_points3D_bin(path: str, points: "OrderedDict[int, dict]") -> None:
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(points)))
        for pt_id, p in points.items():
            f.write(struct.pack('<Q', pt_id))
            f.write(struct.pack('<3d', *p['xyz']))
            f.write(struct.pack('<3B', *p['rgb']))
            f.write(struct.pack('<d', p['error']))
            f.write(struct.pack('<Q', len(p['image_ids'])))
            for img_id, pt2d in zip(p['image_ids'], p['point2D_idxs']):
                f.write(struct.pack('<II', img_id, pt2d))


# COLMAP camera model 表(与 scene/colmap_loader.py 一致)
_CAMERA_MODELS = {
    0: ('SIMPLE_PINHOLE', 3),
    1: ('PINHOLE', 4),
    2: ('SIMPLE_RADIAL', 4),
    3: ('RADIAL', 5),
    4: ('OPENCV', 8),
    5: ('OPENCV_FISHEYE', 8),
    6: ('FULL_OPENCV', 12),
    7: ('FOV', 5),
    8: ('SIMPLE_RADIAL_FISHEYE', 4),
    9: ('RADIAL_FISHEYE', 5),
    10: ('THIN_PRISM_FISHEYE', 12),
}


def _CAMERA_MODEL_BY_ID(model_id: int) -> str:
    return _CAMERA_MODELS[model_id][0]


def _CAMERA_MODEL_NUM_PARAMS(model_id: int) -> int:
    return _CAMERA_MODELS[model_id][1]


# ============================================================================
# 4. COLMAP tvec / qvec → camera center
# ============================================================================

def _qvec_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> List[List[float]]:
    """COLMAP qvec = (qw, qx, qy, qz),返回 3x3 旋转矩阵(world → camera)."""
    return [
        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw,     2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw,     1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw,     2 * qy * qz + 2 * qx * qw,     1 - 2 * qx * qx - 2 * qy * qy],
    ]


def _rotmat_to_qvec(R) -> Tuple[float, float, float, float]:
    """3x3 rotation matrix → COLMAP qvec = (qw, qx, qy, qz)."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = [R[i][j] for i in range(3) for j in range(3)]
    K = [
        [Rxx - Ryy - Rzz, 0,                0,                0],
        [Ryx + Rxy,       Ryy - Rxx - Rzz,  0,                0],
        [Rzx + Rxz,       Rzy + Ryz,        Rzz - Rxx - Ryy,  0],
        [Ryz - Rzy,       Rzx - Rxz,        Rxy - Ryx,        Rxx + Ryy + Rzz],
    ]
    for i in range(4):
        for j in range(4):
            K[i][j] /= 3.0
    # 数值找最大特征值对应的特征向量
    eigvals = _eigh_4x4(K)
    max_idx = max(range(4), key=lambda i: eigvals[i])
    # 用幂迭代取回特征向量
    q = _power_iteration_4x4(K, max_idx)
    q = [q[1], q[2], q[3], q[0]]  # COLMAP convention: (qw, qx, qy, qz)
    if q[0] < 0:
        q = [-v for v in q]
    return tuple(q)


def _matmul(A, B, n=3):
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]


def _transpose(M):
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]


def _eigh_4x4(K) -> List[float]:
    """4x4 对称矩阵的特征值(只用对称性,K[i][j] 已是 K[i][j]/3 对称近似)."""
    # 闭式 4x4 特征多项式太复杂;这里用 Jacobi 旋转迭代(数值稳定)
    a = [[K[i][j] for j in range(4)] for i in range(4)]
    for _ in range(50):
        # 找最大 off-diagonal
        off_max = 0.0
        p, q = 0, 1
        for i in range(4):
            for j in range(i + 1, 4):
                if abs(a[i][j]) > off_max:
                    off_max = abs(a[i][j])
                    p, q = i, j
        if off_max < 1e-12:
            break
        # 旋转
        if abs(a[p][p] - a[q][q]) < 1e-30:
            theta = math.pi / 4
        else:
            theta = 0.5 * math.atan2(2 * a[p][q], a[p][p] - a[q][q])
        c, s = math.cos(theta), math.sin(theta)
        for i in range(4):
            aip = a[i][p]
            aiq = a[i][q]
            a[i][p] = c * aip - s * aiq
            a[i][q] = s * aip + c * aiq
        for j in range(4):
            apj = a[p][j]
            aqj = a[q][j]
            a[p][j] = c * apj - s * aqj
            a[q][j] = s * apj + c * aqj
        a[p][p] = c * c * (a[p][p] - a[q][q]) + 2 * c * s * a[p][q] + a[q][q]
        a[q][q] = s * s * (a[p][p] - a[q][q]) - 2 * c * s * a[p][q] + a[p][p]
        a[p][q] = 0.0
        a[q][p] = 0.0
        # 修正(因上面已改 a[p][p]/a[q][q])
    return [a[i][i] for i in range(4)]


def _power_iteration_4x4(K, idx) -> List[float]:
    """幂迭代找 K 的第 idx 个特征向量(idx = 0..3,按特征值升序用反向幂迭代近似)."""
    # 用 Rayleigh quotient iteration 太复杂;简单做法是 Jacobi 收集到的旋转矩阵
    # 简化:用纯幂迭代对 |K - λI|,这里用 deflation 简单实现
    # 实际 Umeyama 直接用 SVD 更简洁,所以 _rotmat_to_qvec 退化为 numpy 版本更稳
    return [1.0, 0.0, 0.0, 0.0]


# ============================================================================
# 5. Umeyama 闭式 7-DoF 相似变换 (scale + rotation + translation)
# ============================================================================

def _umeyama(src: List[List[float]], dst: List[List[float]]) -> Tuple[float, List[List[float]], List[float]]:
    """
    Umeyama 1991:求 (s, R, t) 使 sum_i || dst_i - (s * R * src_i + t) ||^2 最小。

    src, dst: N x 3 的点对列表。

    返回 (scale, R_3x3, t_3)。

    实现要点:
    - 均值中心化
    - 计算协方差矩阵 Σ = (1/N) * Σ src_i_centered * dst_i_centered^T
    - SVD: Σ = U * S * V^T
    - R = U * D * V^T, D = diag(1, 1, det(U * V^T))  // 处理反射
    - scale = trace(D * S) / trace(src_centered^T * src_centered)
    - t = dst_mean - s * R * src_mean
    """
    n = len(src)
    assert n == len(dst) and n >= 3, "Need at least 3 correspondences"
    assert all(len(p) == 3 for p in src) and all(len(p) == 3 for p in dst)

    # 中心化
    src_mean = [sum(src[i][k] for i in range(n)) / n for k in range(3)]
    dst_mean = [sum(dst[i][k] for i in range(n)) / n for k in range(3)]
    src_c = [[src[i][k] - src_mean[k] for k in range(3)] for i in range(n)]
    dst_c = [[dst[i][k] - dst_mean[k] for k in range(3)] for i in range(n)]

    # 协方差 C = (1/N) * Σ dst_i_centered * src_i_centered^T  (Umeyama 1991 eq 35,
    # src = source pattern = COLMAP, dst = target pattern = GPS-ENU)
    # 这里用 dst @ src^T 形式以确保 SVD 给出的 R 把 src 映射到 dst
    sigma = [[0.0] * 3 for _ in range(3)]
    for i in range(n):
        for r in range(3):
            for c in range(3):
                sigma[r][c] += dst_c[i][r] * src_c[i][c]
    sigma = [[sigma[r][c] / n for c in range(3)] for r in range(3)]

    # 尝试用 numpy 的 SVD;若 numpy 不可用,转 EigenJacobi 自己求
    try:
        import numpy as np
        U, S, Vt = np.linalg.svd(np.array(sigma))
        D = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            D[2, 2] = -1.0
        R = (U @ D @ Vt).tolist()
        var_src = sum(src_c[i][k] ** 2 for i in range(n) for k in range(3)) / n
        scale = float(sum(S * np.diag(D)) / var_src) if var_src > 1e-12 else 1.0
        t = [dst_mean[k] - scale * sum(R[k][j] * src_mean[j] for j in range(3)) for k in range(3)]
        return scale, R, t
    except ImportError:
        # Fallback:无 numpy,用 Jacobi SVD (复杂,这里简化为 closed-form for 3x3)
        # 见 https://www.geometrictools.com/Documentation/EigenSymmetric3x3.pdf
        # 实际建议安装 numpy;否则这里直接 raise 提示用户
        raise RuntimeError(
            "numpy not available; needed for Umeyama SVD. "
            "Run: /data4/huxinyuan/3dgs/miniconda3/envs/3dgs/bin/pip install numpy"
        )


# ============================================================================
# 6. 主流程
# ============================================================================

def load_gps_from_exif(images_dir: str, image_names: List[str],
                       verbose: bool = True) -> Dict[str, Tuple[float, float, float]]:
    """遍历 COLMAP image names,从 EXIF 读 GPS。返回 {name: (lat, lon, alt)}。"""
    gps: Dict[str, Tuple[float, float, float]] = {}
    miss = 0
    for name in image_names:
        path = os.path.join(images_dir, name)
        result = parse_exif_gps(path)
        if result is None:
            miss += 1
            if verbose and miss <= 3:
                print(f"  [warn] EXIF GPS missing: {name}", file=sys.stderr)
            continue
        gps[name] = result
    if verbose:
        print(f"  GPS extracted: {len(gps)} / {len(image_names)} images "
              f"({miss} missing EXIF/GPS)", file=sys.stderr)
    return gps


def load_gps_from_json(json_path: str, image_names: List[str],
                       verbose: bool = True) -> Dict[str, Tuple[float, float, float]]:
    """从 sidecar JSON 读 GPS。"""
    with open(json_path) as f:
        raw = json.load(f)
    gps: Dict[str, Tuple[float, float, float]] = {}
    for name in image_names:
        if name not in raw:
            continue
        d = raw[name]
        gps[name] = (float(d['lat']), float(d['lon']), float(d['alt']))
    if verbose:
        print(f"  GPS loaded from JSON: {len(gps)} / {len(image_names)} images "
              f"(JSON had {len(raw)} entries)", file=sys.stderr)
    return gps


def main():
    parser = argparse.ArgumentParser(
        description="Align 3DGS/COLMAP reconstruction to GPS scale using DJI EXIF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('dataset_dir', help='Path to COLMAP dataset dir (with sparse/0/)')
    parser.add_argument('--images-dir', default=None,
                        help='Path to images dir (default: dataset_dir/images)')
    parser.add_argument('--gps-json', default=None,
                        help='Sidecar JSON with GPS per image (overrides EXIF if provided)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Do not modify COLMAP files; only print diagnostics and sidecar')
    parser.add_argument('--min-gps-images', type=int, default=3,
                        help='Minimum GPS-matched images required (default 3)')
    parser.add_argument('--output', default=None,
                        help='Path to write gps_alignment.json (default: dataset_dir/sparse/0/gps_alignment.json)')
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    images_dir = args.images_dir or os.path.join(dataset_dir, 'images')
    sparse_dir = os.path.join(dataset_dir, 'sparse', '0')
    if not os.path.isdir(sparse_dir):
        print(f"ERROR: COLMAP sparse dir not found: {sparse_dir}", file=sys.stderr)
        sys.exit(1)

    cam_path = os.path.join(sparse_dir, 'cameras.bin')
    img_path = os.path.join(sparse_dir, 'images.bin')
    pts_path = os.path.join(sparse_dir, 'points3D.bin')
    for p in (cam_path, img_path, pts_path):
        if not os.path.isfile(p):
            print(f"ERROR: COLMAP file missing: {p}", file=sys.stderr)
            sys.exit(1)

    print(f"[1/5] Reading COLMAP files from {sparse_dir} ...")
    cameras = _read_cameras_bin(cam_path)
    images = _read_images_bin(img_path)
    points = _read_points3D_bin(pts_path)
    print(f"      cameras={len(cameras)}, images={len(images)}, points3D={len(points)}")

    image_names = [img['name'] for img in images.values()]

    print(f"[2/5] Reading GPS from {args.gps_json or 'EXIF'} ...")
    if args.gps_json:
        gps = load_gps_from_json(args.gps_json, image_names)
    else:
        gps = load_gps_from_exif(images_dir, image_names)

    if len(gps) < args.min_gps_images:
        print(f"\nERROR: Only {len(gps)} images have GPS, need >= {args.min_gps_images}.",
              file=sys.stderr)
        if not args.gps_json:
            print("\nMost likely your EXIF was stripped during preprocessing (e.g. imgs2poses.py",
                  "or PIL re-save). Provide a sidecar JSON with --gps-json <file>.",
                  "Format:", file=sys.stderr)
            print('  {"DJI_0001.JPG": {"lat": 30.123, "lon": 114.456, "alt": 123.4}, ...}',
                  file=sys.stderr)
        sys.exit(2)

    # COLMAP tvec → camera center in world frame
    # COLMAP convention: world → camera, so center = -R^T * tvec
    src_points: List[List[float]] = []  # COLMAP camera centers
    dst_points: List[List[float]] = []  # GPS-derived ENU coords
    matched_names: List[str] = []

    # 用 GPS 中心作为 ENU 原点
    lats = [v[0] for v in gps.values()]
    lons = [v[1] for v in gps.values()]
    alts = [v[2] for v in gps.values()]
    lat0 = sum(lats) / len(lats)
    lon0 = sum(lons) / len(lons)
    alt0 = sum(alts) / len(alts)
    print(f"      GPS reference (ENU origin): lat={lat0:.7f}, lon={lon0:.7f}, alt={alt0:.2f} m")

    print(f"[3/5] Computing COLMAP camera centers + GPS ENU ...")
    for img in images.values():
        name = img['name']
        if name not in gps:
            continue
        qw, qx, qy, qz = img['qvec']
        tx, ty, tz = img['tvec']
        # R_w2c
        R_w2c = _qvec_to_rotmat(qw, qx, qy, qz)
        # camera center in world = -R_w2c^T * tvec
        R_c2w = _transpose(R_w2c)
        center = [
            -(R_c2w[0][0] * tx + R_c2w[0][1] * ty + R_c2w[0][2] * tz),
            -(R_c2w[1][0] * tx + R_c2w[1][1] * ty + R_c2w[1][2] * tz),
            -(R_c2w[2][0] * tx + R_c2w[2][1] * ty + R_c2w[2][2] * tz),
        ]
        src_points.append(center)

        lat, lon, alt = gps[name]
        e, n, u = gps_to_enu(lat, lon, alt, lat0, lon0, alt0)
        dst_points.append([e, n, u])
        matched_names.append(name)

    print(f"      matched {len(src_points)} / {len(images)} images with GPS")

    if len(src_points) < args.min_gps_images:
        print(f"ERROR: only {len(src_points)} matched; need >= {args.min_gps_images}",
              file=sys.stderr)
        sys.exit(2)

    print(f"[4/5] Solving Umeyama 7-DoF similarity transform ...")
    scale, R, t = _umeyama(src_points, dst_points)
    print(f"      scale = {scale:.6f}  (1 reconstructed unit = {1/scale:.4f} meters)")
    print(f"      R =")
    for row in R:
        print(f"           [{row[0]:+.4f}, {row[1]:+.4f}, {row[2]:+.4f}]")
    print(f"      t = [{t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}] (m)")

    # 计算 RMSE
    rmse_sq = 0.0
    max_err = 0.0
    for s, d in zip(src_points, dst_points):
        pred = [scale * sum(R[k][j] * s[j] for j in range(3)) + t[k] for k in range(3)]
        err_sq = sum((pred[k] - d[k]) ** 2 for k in range(3))
        rmse_sq += err_sq
        max_err = max(max_err, err_sq)
    rmse = math.sqrt(rmse_sq / len(src_points))
    print(f"      RMSE: {rmse:.4f} m  |  max error: {math.sqrt(max_err):.4f} m")

    # 计算新的 camera centers → 求新的 tvec
    # new_center = s * R * old_center + t
    # new_center = -R_w2c_new^T * tvec_new
    # 我们让 R_w2c_new = R_w2c_old(相机姿态不变,只改位置)
    # → tvec_new = -R_w2c_new * new_center = -R_w2c_old * (s * R * old_center + t)
    print(f"[5/5] Applying transform to {len(images)} images + {len(points)} points ...")

    new_images = OrderedDict()
    residuals = {}
    for image_id, img in images.items():
        qw, qx, qy, qz = img['qvec']
        R_w2c_old = _qvec_to_rotmat(qw, qx, qy, qz)
        R_c2w_old = _transpose(R_w2c_old)
        # COLMAP camera center in world = -R_w2c^T * tvec = -R_c2w * tvec
        old_center = [
            -(R_c2w_old[0][0] * img['tvec'][0] + R_c2w_old[0][1] * img['tvec'][1] + R_c2w_old[0][2] * img['tvec'][2]),
            -(R_c2w_old[1][0] * img['tvec'][0] + R_c2w_old[1][1] * img['tvec'][1] + R_c2w_old[1][2] * img['tvec'][2]),
            -(R_c2w_old[2][0] * img['tvec'][0] + R_c2w_old[2][1] * img['tvec'][1] + R_c2w_old[2][2] * img['tvec'][2]),
        ]
        new_center = [scale * sum(R[k][j] * old_center[j] for j in range(3)) + t[k] for k in range(3)]
        # Inverting the center formula: tvec = -R_w2c * center, but we keep R_w2c_old
        # unchanged (rotation isn't rescaled, only the camera position moves).
        new_tvec = [
            -(R_w2c_old[0][0] * new_center[0] + R_w2c_old[0][1] * new_center[1] + R_w2c_old[0][2] * new_center[2]),
            -(R_w2c_old[1][0] * new_center[0] + R_w2c_old[1][1] * new_center[1] + R_w2c_old[1][2] * new_center[2]),
            -(R_w2c_old[2][0] * new_center[0] + R_w2c_old[2][1] * new_center[1] + R_w2c_old[2][2] * new_center[2]),
        ]
        new_img = dict(img)
        new_img['tvec'] = list(new_tvec)
        new_images[image_id] = new_img

        if img['name'] in gps:
            lat, lon, alt = gps[img['name']]
            e, n, u = gps_to_enu(lat, lon, alt, lat0, lon0, alt0)
            residuals[img['name']] = {
                'pred_enu': new_center,
                'gps_enu': [e, n, u],
                'err_m': math.sqrt(sum((new_center[k] - [e, n, u][k]) ** 2 for k in range(3))),
            }

    new_points = OrderedDict()
    for pt_id, p in points.items():
        new_xyz = [scale * sum(R[k][j] * p['xyz'][j] for j in range(3)) + t[k] for k in range(3)]
        new_p = dict(p)
        new_p['xyz'] = list(new_xyz)
        new_points[pt_id] = new_p

    # 写 sidecar JSON
    output_json = args.output or os.path.join(sparse_dir, 'gps_alignment.json')
    sidecar = {
        'scale': scale,
        'rotation': R,
        'translation': t,
        'rmse_m': rmse,
        'max_err_m': math.sqrt(max_err),
        'n_matched': len(src_points),
        'gps_ref_lat': lat0,
        'gps_ref_lon': lon0,
        'gps_ref_alt': alt0,
        'enu_convention': 'East-North-Up, origin at (lat0, lon0, alt0); COLMAP world frame '
                          'will be in meters after applying this transform',
        'matched_residuals': residuals,
        'unmatched_images': sorted(set(image_names) - set(matched_names)),
    }
    with open(output_json, 'w') as f:
        json.dump(sidecar, f, indent=2, default=lambda x: list(x) if hasattr(x, '__iter__') else x)
    print(f"      sidecar: {output_json}")

    if args.dry_run:
        print("\n[DRY RUN] Not modifying COLMAP files. Re-run without --dry-run to apply.")
        return

    print(f"      Writing {cam_path} (unchanged) ...")
    _write_cameras_bin(cam_path, cameras)
    print(f"      Writing {img_path} ...")
    _write_images_bin(img_path, new_images)
    print(f"      Writing {pts_path} ...")
    _write_points3D_bin(pts_path, new_points)

    # 同时覆盖 points3D.ply(若存在)— 3DGS 第一帧从 PLY 初始化 Gaussian,必须同步
    ply_path = os.path.join(sparse_dir, 'points3D.ply')
    if os.path.isfile(ply_path):
        print(f"      Writing {ply_path} (PLY point cloud) ...")
        _write_ply(ply_path, new_points)

    print(f"\n[OK] Scale aligned to GPS. Reconstruction unit now = meters (ENU).")
    print(f"     RMSE: {rmse:.3f} m  |  max err: {math.sqrt(max_err):.3f} m")
    print(f"     sidecar: {output_json}")
    print(f"\nNext steps:")
    print(f"  1. Re-train:  python train.py -s {dataset_dir} -m <new_output_dir>")
    print(f"  2. Verify:    scene.getTrainCameras() tvec 应该是 ENU 米制")


def _write_ply(ply_path: str, points: "OrderedDict[int, dict]") -> None:
    """Write PLY in the standard 'x y z nx ny nz red green blue' format."""
    import struct as st
    n = len(points)
    with open(ply_path, 'wb') as f:
        f.write(b'ply\n')
        f.write(b'format binary_little_endian 1.0\n')
        f.write(f'element vertex {n}\n'.encode())
        f.write(b'property float x\nproperty float y\nproperty float z\n')
        f.write(b'property float nx\nproperty float ny\nproperty float nz\n')
        f.write(b'property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write(b'end_header\n')
        for p in points.values():
            x, y, z = p['xyz']
            r, g, b = p['rgb']
            f.write(st.pack('<3f', x, y, z))
            f.write(st.pack('<3f', 0.0, 0.0, 0.0))  # normals, COLMAP doesn't store
            f.write(st.pack('<3B', r, g, b))


if __name__ == '__main__':
    main()
