"""ts_klv2frames package:TS + KLV(MISB ST 0601)→ 抽帧 + EXIF GPS 写入。

模块:
    ts_demux       — MPEG-TS demux
    klv_misb0601   — KLV 解析与编码
    exif_gps       — PNG EXIF GPSInfo 写入(纯 stdlib)
    pipeline       — 编排(抽帧 + KLV 匹配 + 写 PNG)

用法:
    python /home/Pretools/ts_klv2frames.py --ts in.ts --out /tmp/out --frame-interval 13
"""
