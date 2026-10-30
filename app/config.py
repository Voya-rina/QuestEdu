# -*- coding: utf-8 -*-
"""路径与运行期配置。

只放"整个应用共用"的东西：目录位置、应用元信息、局域网地址。
LLM 相关的可变配置在 app/core/llm.py（要从 .env 热加载，不在这里）。
"""
from __future__ import annotations

import socket
from pathlib import Path

# 仓库根目录：app/config.py → 上两级
ROOT = Path(__file__).resolve().parent.parent

APP_DIR = ROOT / "app"
WEB_DIR = APP_DIR / "web"
DATA_DIR = ROOT / "data"          # SQLite 落在这里，已加进 .gitignore
ENV_PATH = ROOT / ".env"          # 页面里保存的 Key 写到这里

APP_NAME = "QuestEdu 学问"
VERSION = "0.2.0"

DATA_DIR.mkdir(exist_ok=True)


def lan_ip() -> str:
    """本机在局域网里的地址，手机扫码/输入都靠它。取不到就回 127.0.0.1。"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"
