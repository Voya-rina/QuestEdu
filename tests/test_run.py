# -*- coding: utf-8 -*-
"""启动器测试。

双击没反应是最难排查的一类问题：用户没有命令行输出可看。所以端口被占、
依赖缺失这些分支必须在代码里有确定行为。
"""
from __future__ import annotations

import socket
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run import pick_port, port_free  # noqa: E402


class TestPorts(unittest.TestCase):
    def test_free_port_is_free(self):
        # 先拿一个系统分配的端口，再释放，基本可以确定它空着
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.assertTrue(port_free("127.0.0.1", port))

    def test_occupied_port_detected(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            port = s.getsockname()[1]
            self.assertFalse(port_free("127.0.0.1", port))

    def test_pick_port_skips_the_busy_one(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            s.listen(1)
            busy = s.getsockname()[1]
            chosen = pick_port("127.0.0.1", busy)
        self.assertNotEqual(chosen, busy)
        self.assertGreater(chosen, busy)


class TestLauncherFiles(unittest.TestCase):
    def test_start_cmd_exists_and_is_ascii(self):
        """cmd.exe 按系统代码页读 .bat，非 ASCII 内容会变乱码。"""
        path = ROOT / "START.cmd"
        self.assertTrue(path.exists())
        data = path.read_bytes()
        try:
            data.decode("ascii")
        except UnicodeDecodeError as e:
            self.fail(f"START.cmd 含非 ASCII 字符：{e}")

    def test_start_cmd_points_to_runner(self):
        text = (ROOT / "START.cmd").read_text(encoding="ascii")
        self.assertIn("scripts\\run.py", text)
        self.assertIn("import fastapi, sympy, uvicorn", text, "必须探到装了依赖的解释器")

    def test_shell_launcher_exists(self):
        path = ROOT / "start.sh"
        self.assertTrue(path.exists())
        self.assertIn("scripts/run.py", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
