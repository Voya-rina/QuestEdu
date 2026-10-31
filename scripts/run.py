# -*- coding: utf-8 -*-
"""开发启动器：起服务、打印局域网地址与二维码、自动打开浏览器。

用法：
    python scripts/run.py                # 监听 8062，自动开浏览器
    python scripts/run.py --port 9000    # 换端口
    python scripts/run.py --no-open      # 不开浏览器
    python scripts/run.py --reload       # 改代码自动重启

Windows 下双击仓库根目录的 `start.cmd` 即可，不用敲命令。
"""
from __future__ import annotations

import argparse
import importlib
import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 运行必需（缺了连启动都启动不了）
REQUIRED = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("sympy", "sympy"),
    ("httpx", "httpx"),
    ("pydantic", "pydantic"),
]


def ensure_deps() -> bool:
    missing = []
    for module, package in REQUIRED:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)
    if not missing:
        return True

    print("\n缺少依赖：" + "、".join(missing))
    print("装一下就好（复制这一行到命令行）：\n")
    print(f'  "{sys.executable}" -m pip install ' + " ".join(missing)
          + " -i https://pypi.tuna.tsinghua.edu.cn/simple\n")
    return False


def port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def ws_backend() -> str:
    """WS 后端有没有装。

    缺了不影响用：前端在 `/ws/chat` 连不上时会自动退回 POST。
    代价是回复变成"整段出现"而不是逐字流出来，控制台还会刷
    "No supported WebSocket library detected"。装上 websockets 即可。

    注意必须真去 import 一次：uvicorn 是在启动时探测的，
    装完以后**已经在跑的进程**不会自己变好，得重启。
    """
    for name in ("websockets", "wsproto"):
        try:
            importlib.import_module(name)
            return name
        except ImportError:
            continue
    return ""


def pick_port(host: str, want: int, tries: int = 20) -> int:
    """端口被占就往后找，别让"双击了没反应"变成一次调试。"""
    for port in range(want, want + tries):
        if port_free(host, port):
            return port
    raise SystemExit(f"{want}~{want + tries - 1} 都被占用了，用 --port 换一个吧")


def dual_stack_socket(port: int):
    """一个监听同时吃 IPv4 和 IPv6。

    Windows 的 IPv6 socket 默认 IPV6_V6ONLY=1：直接绑 `::` 会**只**接 IPv6，
    把原来的 0.0.0.0 弄断。手动把它设成 0 才是真双栈（实测 v4/v6 回环都通）。

    拿不到就算了——回退到只监听 IPv4，不能因为想要 IPv6 把服务弄起不来。
    """
    try:
        s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        s.bind(("::", port))
        return s
    except Exception:
        return None


def print_qr(url: str) -> None:
    try:
        import qrcode  # 可选依赖，没有就只打印地址
    except ImportError:
        print("  （装了 qrcode 这里会打印二维码：pip install qrcode）")
        return
    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.make(fit=True)
    qr.print_ascii(invert=True)


def minimize_console() -> None:
    """Web 已经在浏览器里打开了，把黑窗收进任务栏（最小化，不是隐藏——
    停服务要 Ctrl+C，出错信息也得有地方看，所以不能真藏起来）。"""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 6)      # SW_MINIMIZE
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="QuestEdu 学问 开发服务器")
    parser.add_argument("--host", default="0.0.0.0")
    # 托管沙盒（WorkBuddy / 容器）只给一个 PORT 环境变量，不传 --port
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT") or 8062))
    parser.add_argument("--reload", action="store_true", help="改代码自动重启")
    parser.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    if not ensure_deps():
        raise SystemExit(1)

    from app.config import APP_NAME, VERSION  # 依赖检查之后才能导入
    from app.core.access import find_cloudflared, ipv6_global, lan_ips

    port = pick_port(args.host, args.port)
    if port != args.port:
        print(f"端口 {args.port} 被占用，改用 {port}")

    ips = lan_ips()
    local = f"http://127.0.0.1:{port}"
    lan = f"http://{ips[0]}:{port}" if ips else local
    v6 = ipv6_global()

    print(f"\n{APP_NAME} v{VERSION}")
    print(f"  本机  {local}")
    print(f"  手机  {lan}   ← 同一 WiFi 直接打开")
    # 校园网会换 IP、还可能隔离客户端，把网卡上的地址都列出来让人挨个试
    for ip in ips[1:]:
        print(f"         http://{ip}:{port}   （备用网卡）")
    if v6:
        print(f"  手机  http://[{v6}]:{port}")
        print("         ↑ 有 IPv6 的网络（4G/5G 基本都有）不在同一 WiFi 也能开，第一次要输访问码")
    print("  扫码 ↓")
    print_qr(lan)
    if not ws_backend():
        print("\n提示：没装 websockets，流式通道会降级成一次性返回（功能不受影响）。")
        print(f'  要逐字显示就装一下："{sys.executable}" -m pip install websockets')
    if not find_cloudflared():
        print("\n提示：没找到 cloudflared，公网隧道那条路暂时开不了（局域网与托管不受影响）。")
        print("  下载：https://github.com/cloudflare/cloudflared/releases/latest/download/"
              "cloudflared-windows-amd64.exe")
        print("  放到 E:\\AI\\cloudflared\\cloudflared.exe 后重启本服务即可。")
    print("\n按 Ctrl+C 停止\n")

    if not args.no_open:
        # 开浏览器顺带把控制台窗口最小化：页面都开了，黑窗没必要继续占地方
        threading.Timer(1.2, lambda: (webbrowser.open(local), minimize_console())).start()

    import uvicorn

    # 0.0.0.0 时改用双栈 socket：手机走 IPv6 也能直接连，不靠任何穿透工具。
    # --reload 由 watchfiles 驱动、自带 bind 逻辑，就不掺和了。
    sock = None
    if not args.reload and args.host in ("0.0.0.0", "::"):
        sock = dual_stack_socket(port)
    if sock is not None:
        print("  已同时监听 IPv4 与 IPv6（同一个端口）")
        uvicorn.Server(uvicorn.Config("app.main:app", log_level="warning")).run(sockets=[sock])
    else:
        uvicorn.run("app.main:app", host=args.host, port=port,
                    reload=args.reload, log_level="warning")


if __name__ == "__main__":
    main()
