"""job_id 命名约定 回归测试。

job_id 格式:
- 有 name:`YYYYMMDDHHMMSS-<sanitized_name>-<4hex>`
  例: `20260904163242-my_scene-a1b2`
- 没 name:`YYYYMMDDHHMMSS-<8hex>`(旧行为兼容)

sanitize 规则:
- 转小写
- 把连续非 [a-z0-9_.-] 字符替换为单个 _
- 去掉前导 / 尾随的 _ - .
- 截断到 40 字符
- 全特殊字符 → 空串(走 fallback)
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

_API_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_API_DIR))

from pipeline3dgs.routes import _sanitize_job_name  # noqa: E402


class TestSanitizeJobName(unittest.TestCase):

    def test_lowercase(self):
        self.assertEqual(_sanitize_job_name("MyScene"), "myscene")
        self.assertEqual(_sanitize_job_name("CAMPUS"), "campus")

    def test_keeps_alphanumeric_underscore_dash_dot(self):
        self.assertEqual(_sanitize_job_name("my_scene-v1.2"), "my_scene-v1.2")
        self.assertEqual(_sanitize_job_name("abc123"), "abc123")

    def test_replaces_special_chars_with_underscore(self):
        # 中文 → 全替成 _,然后 leading _ 被 strip
        self.assertEqual(_sanitize_job_name("孙村10kV"), "10kv")
        # 空格 → _,无前后 _ → 保留 _
        self.assertEqual(_sanitize_job_name("hello world"), "hello_world")
        # 连续特殊字符 → 单个 _
        self.assertEqual(_sanitize_job_name("a!!b??c"), "a_b_c")
        # 路径分隔符 → _ (防目录注入)
        self.assertEqual(_sanitize_job_name("../etc/passwd"), "etc_passwd")
        self.assertEqual(_sanitize_job_name("a/b\\c"), "a_b_c")

    def test_strips_leading_trailing_punctuation(self):
        self.assertEqual(_sanitize_job_name("___name___"), "name")
        self.assertEqual(_sanitize_job_name("..hello.."), "hello")
        self.assertEqual(_sanitize_job_name("-_-_scene_-_-"), "scene")

    def test_truncates_long_names(self):
        long = "a" * 100
        result = _sanitize_job_name(long)
        self.assertEqual(len(result), 40)
        self.assertEqual(result, "a" * 40)

    def test_truncation_strips_trailing_punct(self):
        # 截断到 40 后,尾字符若正好是 _/-/. 再 rstrip
        result = _sanitize_job_name("a" * 39 + "-")  # 总长 40,第 40 字符是 -
        self.assertEqual(result, "a" * 39)  # 尾随 - 被 rstrip

    def test_empty_inputs_return_empty(self):
        self.assertEqual(_sanitize_job_name(None), "")
        self.assertEqual(_sanitize_job_name(""), "")
        self.assertEqual(_sanitize_job_name("   "), "")  # strip 后空
        self.assertEqual(_sanitize_job_name("___"), "")  # 全特殊字符
        self.assertEqual(_sanitize_job_name("---"), "")
        self.assertEqual(_sanitize_job_name("..."), "")

    def test_unicode_normalized_to_underscore(self):
        # 中文 / emoji / 阿拉伯文 都被认为是"非安全字符" → _,然后 strip 掉末尾 _
        # (我们不做 unicode NFKC,因为可能会意外拼长字符串)
        self.assertEqual(_sanitize_job_name("scene_🎬"), "scene")
        self.assertEqual(_sanitize_job_name("café"), "caf")  # é → _,末尾 strip
        # 注意:实际用户不太可能输入这些,但若输入了,结果可预测

    def test_filename_like_input(self):
        # 模拟 fastapi UploadFile.filename("孙村10kV智能线-2 2026.9.2/1.平滑S形前进...MP4")
        # 中文 + 路径分隔符全替成 _;尾随 ... 被 strip;中段 _- 保留
        self.assertEqual(
            _sanitize_job_name("孙村10kV智能线-2 2026.9.2/1.平滑S形前进...MP4"),
            "10kv_-2_2026.9.2_1._s_...mp4",
        )

    def test_typical_user_inputs(self):
        cases = {
            "scene": "scene",
            "scene_01": "scene_01",
            "campus": "campus",
            "biandianzhan7": "biandianzhan7",
            "9_1_1_low": "9_1_1_low",
            "9_1_2": "9_1_2",
            "Drone-A 2026-09-04": "drone-a_2026-09-04",
        }
        for inp, expected in cases.items():
            with self.subTest(input=inp):
                self.assertEqual(_sanitize_job_name(inp), expected)


class TestJobIdFormat(unittest.TestCase):
    """实际构造 job_id 字符串,验证最终格式(不实际调 API)。"""

    def _build_job_id(self, ts: str, name: str | None, hex_tail: str) -> str:
        """镜像 routes.py:174-180 逻辑。"""
        safe_name = _sanitize_job_name(name)
        if safe_name:
            return f"{ts}-{safe_name}-{hex_tail[:4]}"
        return f"{ts}-{hex_tail[:8]}"

    def test_with_name_format(self):
        jid = self._build_job_id("20260904163242", "my_scene", "a1b2c3d4e5f6")
        self.assertEqual(jid, "20260904163242-my_scene-a1b2")
        # 字典序可排序
        self.assertLess(jid, "20260904163242-my_scene-z9z9")

    def test_without_name_falls_back_to_old_format(self):
        jid = self._build_job_id("20260904163242", None, "a1b2c3d4e5f6")
        self.assertEqual(jid, "20260904163242-a1b2c3d4")

    def test_with_chinese_name(self):
        # 实际用户的视频名,带中文 + 路径分隔符
        jid = self._build_job_id(
            "20260904163242",
            "孙村10kV智能线-2 2026.9.2/1.平滑S形前进...MP4",
            "deadbeef",
        )
        self.assertEqual(jid, "20260904163242-10kv_-2_2026.9.2_1._s_...mp4-dead")

    def test_unique_at_second_granularity(self):
        # 同秒内同 name 仍能区分(4hex 后缀 = 65536 种)
        jid_a = self._build_job_id("20260904163242", "scene", "00000000")
        jid_b = self._build_job_id("20260904163242", "scene", "ffffffff")
        self.assertNotEqual(jid_a, jid_b)


if __name__ == "__main__":
    unittest.main()