"""MPEG-TS demux:从 TS 流提取 KLV PES payload。

参考:
  - ISO/IEC 13818-1 (MPEG-TS)
  - SMPTE ST 336 (KLV)
  - MISB ST 0601 (UAS Datalink Local Set,KLV 嵌入 TS 私有流)

典型 TS 包结构 (188 字节):
    0x47  TEI|PUS|priority  PID_hi  PID_lo  AFC  [adaptation]  [payload]

KLV 在 TS 里的常见携带方式:
  - 私有 PES 流(PID 在 0x0100~0x0FFF 或 0x100~0x1FFF 范围,具体由 mux 配置决定)
  - PES packet_data_byte 里直接放 KLV triple(UDS)

本模块职责:从 TS 文件流式读取,组装 PES packet,吐出 PES payload。
由调用方负责解析 payload 里的 KLV。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterator

# TS 常量
TS_PACKET_SIZE = 188
SYNC_BYTE = 0x47
STREAM_ID_PRIVATE_1 = 0xBD  # private_stream_1 — KLV 经常走这里


@dataclass
class TsPacket:
    pid: int
    pusi: bool               # payload_unit_start_indicator
    adaptation_field: int    # 0=none, 1=only, 2=both, 3=reserved
    payload: bytes


@dataclass
class PesAssembler:
    """PES packet 组装器:跨多个 TS 包拼成一个 PES payload。

    用法:
        asm = PesAssembler()
        for pkt in iter_packets(f):
            for pes in asm.feed(pkt):
                yield pes   # pes = {"stream_id": int, "payload": bytes}
    """
    _state: int = 0            # 0=找 PUSI,1=收 PES header,2=收 payload
    _buf: bytearray = field(default_factory=bytearray)
    _stream_id: int = 0

    def feed(self, pkt: TsPacket) -> Iterator[dict]:
        if self._state == 0:
            if not pkt.pusi or not pkt.payload:
                return
            # PES start_code_prefix + stream_id = 4 bytes
            if len(pkt.payload) < 4:
                return
            if pkt.payload[0:3] != b"\x00\x00\x01":
                return
            self._stream_id = pkt.payload[3]
            self._buf.extend(pkt.payload)
            self._state = 1
        elif self._state == 1:
            self._buf.extend(pkt.payload)
        elif self._state == 2:
            self._buf.extend(pkt.payload)

        if self._state == 1:
            # 解析 PES header: 长度字段 + flags
            # 00 00 01 SID  PES_packet_length(2)  [PES header data]
            if len(self._buf) < 9:
                return
            pes_pkt_len = (self._buf[4] << 8) | self._buf[5]
            # header_data_length = self._buf[8]
            header_len = 9 + self._buf[8]
            # 如果 pes_pkt_len == 0,表示长度未定,读到下一个 PUSI 为止
            if pes_pkt_len == 0:
                self._state = 2
            else:
                # PES packet 总长度 = PES header(6) + PES_packet_length
                # (PES header data + payload 都在 PES_packet_length 内)
                total_len = 6 + pes_pkt_len
                if len(self._buf) >= total_len:
                    payload = bytes(self._buf[header_len:total_len])
                    self._buf.clear()
                    self._state = 0
                    yield {"stream_id": self._stream_id, "payload": payload}
                # else 继续收(后续非 PUSI TS 包)
        elif self._state == 2:
            # 长度未定模式:读到下一个 PUSI 才结算
            # 这里简化:返回当前累积(配合外层按 PUSI 切分)
            payload = bytes(self._buf[9:])  # 跳过 PES header
            self._buf.clear()
            self._state = 0
            yield {"stream_id": self._stream_id, "payload": payload, "unbounded": True}


def iter_packets(f: BinaryIO) -> Iterator[TsPacket]:
    """流式读取 TS 包(188 字节/包)。"""
    while True:
        data = f.read(TS_PACKET_SIZE)
        if len(data) < TS_PACKET_SIZE:
            return
        if data[0] != SYNC_BYTE:
            # 重新对齐 sync byte(容忍错位)
            # 不在循环里无限找;一次性跳到下一个 0x47
            next_off = data.find(b"\x47", 1)
            if next_off < 0:
                continue
            data = data[next_off:] + f.read(TS_PACKET_SIZE - next_off)
            if len(data) < TS_PACKET_SIZE or data[0] != SYNC_BYTE:
                continue
        pusi = (data[1] & 0x40) != 0
        pid = ((data[1] & 0x1F) << 8) | data[2]
        if pid == 0x1FFF:
            continue  # null packet
        afc = (data[3] >> 4) & 0x03
        idx = 4
        if afc in (2, 3):  # adaptation_field present
            af_len = data[idx]
            idx += 1 + af_len
        if afc in (1, 3):
            payload = bytes(data[idx:])
        else:
            payload = b""
        yield TsPacket(pid=pid, pusi=pusi, adaptation_field=afc, payload=payload)


MISB_ST0601_UL = bytes.fromhex("060E2B34020B01010E01030101000000")


def _looks_like_misb0601(payload: bytes) -> bool:
    """检查 payload 前几个字节是不是 MISB ST 0601 UL。

    标准 PES header 9 字节,简化 PES 6 字节;PES payload 里嵌入 KLV 时
    KLV 从某个固定偏移开始。我们只看 PES payload 头 32 字节内能否找到 UL。
    """
    ul = MISB_ST0601_UL
    head = payload[:64]
    return ul in head


def demux_klv_payload(ts_path: Path, pid: int | None = None) -> Iterator[bytes]:
    """从 TS 文件提取 KLV PES payload。

    pid:
      None  — 自动找第一个看起来像 KLV(MISB ST 0601 UL)的私有流
      int   — 指定 PID(已知 mux 配置时用)

    Yields:
        KLV payload bytes(PES payload,内容应为 KLV triple 序列)
    """
    with ts_path.open("rb") as f:
        asm_by_pid: dict[int, PesAssembler] = {}

        # 自动探测 PID 模式:第一次见到 PUSI 时,如果是 private_stream_1 就锁定
        auto_pid: int | None = pid
        if pid is None:
            # 先扫一次找到 PID
            for pkt in iter_packets(f):
                if pkt.pusi and pkt.payload[:4] == b"\x00\x00\x01":
                    sid = pkt.payload[3]
                    if sid == STREAM_ID_PRIVATE_1:
                        auto_pid = pkt.pid
                        break
            if auto_pid is None:
                return
            # 重置文件指针(从头再来)
            f.seek(0)
            asm_by_pid = {auto_pid: PesAssembler()}

        elif pid in (0x0102, 0x0211) or (0x100 <= pid <= 0x1FFF):
            asm_by_pid[pid] = PesAssembler()
        else:
            return

        # 第二遍:组装 PES payload,直接 yield(避免 generator-in-generator)
        results: list[bytes] = []
        for pkt in iter_packets(f):
            if pkt.pid != auto_pid:
                continue
            asm = asm_by_pid[auto_pid]
            _collect_pes_from_asm(asm, pkt, results)
        for payload in results:
            yield payload


def _collect_pes_from_asm(asm: "PesAssembler", pkt: "TsPacket",
                          out: list[bytes]) -> None:
    """把 PesAssembler.feed 的 yield 收集到 out 列表(避免 generator-nested 语义差异)。"""
    for pes in asm.feed(pkt):
        if pes.get("stream_id") == STREAM_ID_PRIVATE_1:
            out.append(pes["payload"])
