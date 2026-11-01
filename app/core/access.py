# -*- coding: utf-8 -*-
"""访问入口：局域网直连 / Cloudflare 隧道 / WorkBuddy 托管，外加一道访问码。

为什么要有访问码：后两种入口都是"把本机服务暴露到公网"。链接一旦泄露，
陌生人就能消耗你的模型额度、翻学生的做题记录。这里不做账号体系（那是另一件事），
只做一道能挡住路人的门：4 位数字 + cookie。

**谁需要输码，按 Host 判断，不按来源 IP。**
cloudflared 就跑在本机，它转发进来的请求 RemoteAddr 是 127.0.0.1——按 IP 判断
等于对所有隧道流量放行。所以只认 Host：127.0.0.1 / localhost / 本机各网卡地址
算"自己人"，trycloudflare 域名、托管域名一律拦。
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import shutil
import socket
import subprocess
import threading
import time
from collections import deque
from urllib.parse import parse_qs, urlparse

from app.config import ROOT, WEB_DIR, lan_ip

log = logging.getLogger(__name__)

CONFIG_PATH = ROOT / ".access.json"      # 已加进 .gitignore，和 .env 同级
COOKIE = "qe_pass"
CODE_LEN = 4

_LAN_WHITELIST = {
    "localhost", "127.0.0.1", "::1", "[::1]", "0.0.0.0",
    # 测试客户端：TestClient 的 Host 固定是 testserver，不放行会让所有接口测试 401
    "testserver",
}

_TUNNEL_URL = re.compile(r"https://[a-z0-9][a-z0-9-]*\.trycloudflare\.com")
# natapp / cpolar 的免费域名都带自家关键字，按关键字认——比穷举 TLD 稳
_TUNNEL_URL_NATAPP = re.compile(r"https://[a-z0-9-]+\.[a-z0-9.-]*natapp[a-z0-9-]*\.[a-z]{2,}")
_TUNNEL_URL_CPOLAR = re.compile(r"https://[a-z0-9-]+\.[a-z0-9.-]*cpolar[a-z0-9-]*\.[a-z]{2,}")
_TUNNEL_FAIL = re.compile(r"(failed to request quick Tunnel|failed to authenticate|"
                          r"context canceled|no such host|TLS handshake|"
                          r"Authtoken|认证失败|登录失败|隧道不)", re.I)

_TUNNEL_PROVIDERS = ("cloudflared", "natapp", "cpolar")


def tunnel_config() -> tuple[str, str]:
    """当前隧道方案（cloudflared / natapp / cpolar）与其 authtoken。"""
    d = _read()
    p = str(d.get("tn_provider") or "cloudflared")
    if p not in _TUNNEL_PROVIDERS:
        p = "cloudflared"
    return p, str(d.get("tn_token") or "")


def set_tunnel_config(provider: str, token: str | None = None) -> tuple[str, str]:
    """保存隧道方案。token 传 None / 空串＝沿用已保存的（密码框惯例）。"""
    provider = (provider or "").strip().lower()
    if provider not in _TUNNEL_PROVIDERS:
        raise AccessError("隧道方案只支持 cloudflared / natapp / cpolar")
    d = _read()
    d["tn_provider"] = provider
    if token:
        d["tn_token"] = str(token).strip()
    _write(d)
    return tunnel_config()


class AccessError(RuntimeError):
    """入口相关的人话错误（前端直接展示）。"""


# ---------------------------------------------------------------- 配置读写

def _read() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write(data: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def code() -> str:
    """当前访问码。没有就生成一个 4 位数字并存下来。"""
    d = _read()
    c = str(d.get("code") or "")
    if len(c) == CODE_LEN and c.isdigit():
        return c
    c = f"{secrets.randbelow(10 ** CODE_LEN):0{CODE_LEN}d}"
    d["code"] = c
    d.setdefault("secret", secrets.token_hex(16))
    _write(d)
    return c


def set_code(new: str) -> str:
    new = (new or "").strip()
    if len(new) != CODE_LEN or not new.isdigit():
        raise AccessError(f"访问码要 {CODE_LEN} 位数字")
    d = _read()
    d["code"] = new
    d.setdefault("secret", secrets.token_hex(16))
    _write(d)
    return new


def host_url() -> str:
    return str(_read().get("host_url") or "")
def set_host_url(url: str) -> str:
    url = (url or "").strip()
    if url and not url.startswith(("http://", "https://")):
        raise AccessError("链接要以 http:// 或 https:// 开头")
    d = _read()
    if url:
        d["host_url"] = url
    else:
        d.pop("host_url", None)
    _write(d)
    return url


# ---------------------------------------------------------------- 演示模式

def demo_mode() -> bool:
    """演示模式：开着时，模型设置（/api/llm/config、/test）服务端直接拒改。

    防的是答辩/演示时底下观众扫码进来乱动设置——光靠前端把输入框灰掉
    不算数，POST 可以绕过，必须在服务端拦。
    """
    return bool(_read().get("demo"))


def set_demo(on: bool, ip: str = "") -> None:
    d = _read()
    d["demo"] = bool(on)
    if on:
        d["demo_ip"] = ip            # 记下开启者：只有这台机器能解锁
    else:
        d.pop("demo_ip", None)
    _write(d)


def demo_ip() -> str:
    return str(_read().get("demo_ip") or "")


# ---------------------------------------------------------------- 控制端识别

def normalize_peer(ip: str) -> str:
    """把 IPv4-mapped IPv6（::ffff:127.0.0.1）还原成 IPv4 写法。

    服务按双栈绑在 `::` 上（见 scripts/run.py 的 dual_stack_socket），所有
    IPv4 连接的对端地址都带 ::ffff: 前缀。不还原的话，控制端识别
    （对端是不是本机）在 IPv4 访问下全部失效——演示者自己的电脑也会被
    判成观众设备，设置/演示模式统统 403。实测踩过。
    """
    p = (ip or "").strip().lower().strip("[]")
    if p.startswith("::ffff:") and "." in p:
        return p[len("::ffff:"):]
    return p


def is_controller(peer: str, headers) -> bool:
    """这台设备是不是控制端（一对多里的"一"）。

    这个产品天然一对多：跑服务的电脑是控制端（演示者），扫码进来的
    全是用户端（学生/观众）。权限边界就画在这里——

    - 控制端：socket 对端 IP 是本机地址（loopback 或本机网卡）；
    - **不认转发头**：cloudflared 也跑在本机，隧道流量的对端同样是
      127.0.0.1，但它必然带着转发头（needs_code 已用它区分内外），
      有转发头一律按用户端处理，否则隧道观众全能冒充控制端；
    - 局域网里其他设备（包括同 WiFi 的观众手机）对端 IP 是它们自己，
      天然不是控制端。
    """
    for h in _FORWARD_HEADERS:
        if headers.get(h):
            return False
    p = normalize_peer(peer)
    if p == "testclient":                  # FastAPI TestClient：单元测试视为控制端
        return True
    if p in ("127.0.0.1", "::1", "localhost"):
        return True
    return p in {ip.lower() for ip in lan_ips()}


# ---------------------------------------------------------------- 访问码校验

def _token(c: str) -> str:
    secret = (_read().get("secret") or "").encode()
    return hmac.new(secret, c.encode(), hashlib.sha256).hexdigest()[:24]


def cookie_value() -> str:
    return _token(code())


def cookie_ok(value: str | None) -> bool:
    if not value:
        return False
    return hmac.compare_digest(value, cookie_value())


def is_local_host(host: str) -> bool:
    """这个 Host 是不是"自己人"（本机 / 局域网 / 测试客户端）。"""
    h = (host or "").strip().lower()
    if h.startswith("["):                 # [::1]:8062 这种 IPv6 写法
        h = h[1:h.index("]")] if "]" in h else h[1:]
    else:
        h = h.split(":")[0]
    if not h:
        return True                       # 没有 Host 头（HTTP/1.0、探活）不拦
    if h in _LAN_WHITELIST:
        return True
    if h in {ip.lower() for ip in lan_ips()}:
        return True
    if h.endswith(".local") or h.endswith(".lan"):
        return True
    return False


# 反代一定会加这些头。别只看 Host：`curl -H "Host: 127.0.0.1" https://xxx.trycloudflare.com`
# 就能把 Host 伪造成"自己人"，可转发头是边缘注入的，伪造不了（伪造只会让自己多输一次码）
_FORWARD_HEADERS = ("cf-connecting-ip", "x-forwarded-for", "x-real-ip",
                    "x-forwarded-host", "cf-ray")


def needs_code(host: str, headers) -> bool:
    """这次请求要不要过访问码。

    两个条件任一成立就要：来了转发头（说明是从隧道/托管那边绕进来的），
    或者 Host 不是本机/局域网地址。
    """
    for h in _FORWARD_HEADERS:
        if headers.get(h):
            return True
    return not is_local_host(host)


# 粗暴但够用的限流：4 位数字只有 1 万种，不能让人随便试
_attempts: dict[str, list[float]] = {}
_MAX_TRIES = 8          # 每分钟
_attempts_lock = threading.Lock()


def too_many_tries(ip: str) -> bool:
    now = time.time()
    with _attempts_lock:
        keys = [t for t in _attempts.get(ip, []) if now - t < 60]
        _attempts[ip] = keys
        return len(keys) >= _MAX_TRIES


def note_try(ip: str) -> None:
    with _attempts_lock:
        _attempts.setdefault(ip, []).append(time.time())


# ---------------------------------------------------------------- 局域网地址

def lan_ips() -> list[str]:
    """本机所有能用的局域网 IPv4，默认路由那个排第一。

    校园网里 IP 会变，把候选都列出来总比只显示一个强——换地址后
    手机还可以试另一个（比如电脑热点的 192.168.x.x）。
    """
    out: list[str] = []
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if ip.startswith(("127.", "169.254.")):
                continue
            if ip not in out:
                out.append(ip)
    except Exception:
        pass
    primary = lan_ip()
    if primary and primary != "127.0.0.1":
        out = [primary] + [ip for ip in out if ip != primary]
    return out


def ipv6_global() -> str:
    """本机可用的公网 IPv6（出网时用的那一个）。

    取法和 lan_ip() 一样：UDP connect 一个外部地址，看内核选了哪个源地址。
    不用 hostname 解析——Windows 上那通常只回 `::1`。

    校园网会通过 RA 发公网前缀，所以这里经常真有地址。注意 Windows 默认
    用临时地址（Privacy Extension），隔几小时就变；临时开着给手机看够用，
    界面上会提醒一句。
    """
    for probe in ("2400:3200::1", "2400:da00::6666", "2001:4860:4860::8888"):
        try:
            s = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
            s.connect((probe, 53))
            ip = s.getsockname()[0]
            s.close()
        except Exception:
            continue
        if not ip or ip in ("::1", "::", "0.0.0.0"):
            continue
        low = ip.lower()
        # 链路本地 / 唯一本地地址：外面回不来，给了也没用
        if low.startswith(("fe80", "fc", "fd")):
            continue
        return ip
    return ""


def safe_qr_target(url: str) -> bool:
    """二维码只编自家地址，别让人把它当免费二维码生成器用。"""
    try:
        p = urlparse(url)
    except Exception:
        return False
    if p.scheme not in ("http", "https"):
        return False
    host = (p.hostname or "").lower()
    if not host:
        return False
    if host in _LAN_WHITELIST or host in {ip.lower() for ip in lan_ips()}:
        return True
    v6 = ipv6_global().lower()
    if v6 and host == v6:
        return True
    if host.endswith(".trycloudflare.com") or host.endswith(".workbuddy.host"):
        return True
    return False


# ---------------------------------------------------------------- Cloudflare 隧道

def find_cloudflared() -> str:
    r"""找 cloudflared。顺序：环境变量 → 项目根目录 Cloudflared\ → PATH → 常见位置。

    实际文件名可能带平台后缀（cloudflared-windows-amd64.exe），两种都认。
    """
    env = os.environ.get("CLOUDFLARED") or ""
    if env and os.path.isfile(env):
        return env
    names = ("cloudflared.exe", "cloudflared-windows-amd64.exe")
    candidates: list[str] = []
    for name in names:
        candidates.append(str(ROOT / "Cloudflared" / name))       # 项目自带（已 gitignore）
    for name in names:
        candidates.append(str(ROOT.parent / "Cloudflared" / name))
    candidates += [
        r"E:\AI\cloudflared\cloudflared.exe",
        r"C:\Program Files\cloudflared\cloudflared.exe",
        r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    found = shutil.which("cloudflared")
    return found or ""


def find_tunnel_bin(provider: str) -> str:
    r"""找指定方案的可执行文件。natapp / cpolar 和 cloudflared 同一套搜索路径，
    放进项目根目录的 Cloudflared\ 文件夹（或 PATH）即可。"""
    if provider == "cloudflared":
        return find_cloudflared()
    env = os.environ.get("TUNNEL_BIN") or ""
    if env and os.path.isfile(env):
        return env
    names = (f"{provider}.exe", f"{provider}-windows-amd64.exe")
    candidates: list[str] = []
    for base in (ROOT / "Cloudflared", ROOT / "Tunnel", ROOT.parent / "Cloudflared"):
        for name in names:
            candidates.append(str(base / name))
    candidates += [fr"E:\AI\cloudflared\{name}" for name in names]
    candidates += [fr"C:\Program Files\{name}" for name in names]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return shutil.which(provider) or ""


_BIN_MISSING = {
    "cloudflared": "没找到 cloudflared。把它放到项目根目录的 Cloudflared\\ 下"
                   "（cloudflared.exe 或 cloudflared-windows-amd64.exe），重启服务即可",
    "natapp": "没找到 natapp。去 natapp.cn 下载客户端，改名 natapp.exe 放进项目根目录的"
              " Cloudflared\\ 文件夹；并在官网创建 web 隧道（指向 127.0.0.1:本服务端口）拿 authtoken",
    "cpolar": "没找到 cpolar。去 cpolar.com 下载 Windows 客户端，把 cpolar.exe 放进项目根目录的"
              " Cloudflared\\ 文件夹，并在官网拿到 authtoken",
}


class Tunnel:
    """一条公网隧道的生命周期，支持三个方案：

    - cloudflared quick tunnel：免登录，代价是每次重启换域名、且对网络环境挑剔
      （默认走 QUIC/UDP，校园网掐 UDP 就频繁断流——所以固定 `--protocol http2`
      强制走 TCP，实测在掐 UDP 的网络里稳得多）
    - natapp：国内节点，低延迟；需要官网注册 + authtoken，本地端口在官网配
    - cpolar：同上，authtoken 首次写入它的配置，之后直接起
    """

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.url = ""
        self.state = "off"          # off | starting | on | error
        self.message = ""
        self.provider = "cloudflared"
        self._lock = threading.Lock()
        self._log: deque = deque(maxlen=80)   # 隧道进程的输出尾部，排错用

    # -- 状态 ------------------------------------------------------------
    def snapshot(self) -> dict:
        alive = self.proc is not None and self.proc.poll() is None
        if self.state in ("on", "starting") and not alive:
            self.state = "error"
            self.message = self.message or "隧道进程已退出"
        provider, token = tunnel_config()
        return {
            "state": self.state,
            "url": self.url if self.state == "on" else "",
            "message": self.message,
            "binary": find_tunnel_bin(provider),
            "provider": provider,
            "has_token": bool(token) or provider == "cloudflared",
        }

    # -- 启动 ------------------------------------------------------------
    def start(self, port: int, wait: float = 30.0) -> dict:
        provider, token = tunnel_config()
        self.provider = provider
        with self._lock:
            if self.proc is not None and self.proc.poll() is None:
                if self.state in ("starting", "on"):
                    return self.snapshot()      # 已经在起/已起，别再起一个
                self.stop()

            exe = find_tunnel_bin(provider)
            if not exe:
                self.state, self.message = "error", _BIN_MISSING[provider]
                return self.snapshot()
            if provider in ("natapp", "cpolar") and not token:
                self.state, self.message = "error", (
                    f"{provider} 需要 authtoken：在官网注册后创建隧道（指向 "
                    f"127.0.0.1:{port}），把 token 填进控制台的 Token 框")
                return self.snapshot()

            if provider == "natapp":
                # natapp 的本地端口在官网隧道配置里填，命令行只给 token；
                # 日志走 stderr，已被 STDERR=STDOUT 合并捕获，不用加 log 参数
                cmd = [exe, f"-authtoken={token}"]
            elif provider == "cpolar":
                # authtoken 先写进它的配置文件（一条快命令），再起隧道
                try:
                    subprocess.run([exe, "authtoken", token],
                                   capture_output=True, timeout=15,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                except Exception:
                    pass                    # 写失败不拦路：它可能已经配过了
                cmd = [exe, "http", str(port)]
            else:
                cmd = [exe, "tunnel", "--url", f"http://127.0.0.1:{port}",
                       "--no-autoupdate", "--metrics", "127.0.0.1:0",
                       "--protocol", "http2"]

            self.state, self.url, self.message = "starting", "", ""
            self._log.clear()
            flags = 0
            if os.name == "nt":
                flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            try:
                self.proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace",
                    creationflags=flags, bufsize=1,
                )
            except Exception as e:
                self.state, self.message = "error", f"启动失败：{e}"
                return self.snapshot()

            threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()

        deadline = time.time() + wait
        while time.time() < deadline:
            if self.state in ("on", "error"):
                break
            time.sleep(0.2)
        if self.state == "starting":
            if self.proc.poll() is None:
                # 首次申请域名实测 4 秒，慢的话多半是网络抖动——
                # 进程还活着就保持 starting，让前端轮询，别急着报错
                self.message = "还没拿到域名，再等几秒；不行就关掉开关重开"
            else:
                self.state = "error"
                self.message = self.message or "隧道进程启动后退出了"
        return self.snapshot()

    def _pump(self, proc: subprocess.Popen) -> None:
        """读隧道进程的输出，捞出公网地址；尾部留存用于排错。"""
        pat = {"cloudflared": _TUNNEL_URL,
               "natapp": _TUNNEL_URL_NATAPP,
               "cpolar": _TUNNEL_URL_CPOLAR}[self.provider]
        try:
            for line in proc.stdout or []:
                line = line.rstrip()
                if line:
                    self._log.append(line)
                m = pat.search(line)
                if m and self.state != "on":
                    self.url = m.group(0)
                    self.state = "on"
                    self.message = ""
                    continue
                if _TUNNEL_FAIL.search(line) and self.state == "starting":
                    self.state = "error"
                    self.message = line.strip()[:200]
        except Exception:
            pass
        if self.state == "starting" and proc.poll() is not None:
            self.state = "error"
            self.message = self.message or "隧道进程启动后立刻退出了"
            tail = " | ".join(list(self._log)[-2:])
            if tail:
                self.message += "：" + tail[:220]

    # -- 停止 ------------------------------------------------------------
    def stop(self) -> dict:
        p, self.proc = self.proc, None
        self.url = ""
        if p is not None and p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=3)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        if self.state != "error":
            self.state = "off"
        return self.snapshot()


tunnel = Tunnel()


def describe(port: int, demo_unlockable: bool = True,
             controller: bool = True) -> dict:
    ips = lan_ips()
    v6 = ipv6_global()
    return {
        "lan": {
            "ips": ips,
            "url": f"http://{ips[0]}:{port}" if ips else "",
            "candidates": [f"http://{ip}:{port}" for ip in ips],
        },
        # 公网 IPv6：不经过任何第三方，代价是地址会变、且第一次要输访问码
        "ipv6": {
            "ip": v6,
            "url": f"http://[{v6}]:{port}" if v6 else "",
        },
        "tunnel": tunnel.snapshot(),
        "host": {"url": host_url()},
        "controller": bool(controller),
        # 演示模式：unlockable 只告诉"该解锁的那台机器"，别的设备连按钮都不给
        "demo": {"on": demo_mode(), "unlockable": bool(demo_unlockable)},
        # 访问码只发给控制端——观众不该看到码（看到就能转发给别人）。
        # 演示模式中控制端**仍然能看到**：演示者要把码私下发给要进来的人，
        # 全打码的话演示中根本没人连得上（实测踩过）
        "gate": {"code": code() if controller else "····",
                 "cookie": COOKIE, "length": CODE_LEN},
    }


# ---------------------------------------------------------------- 门（ASGI 中间件）

GATE_PAGE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#F5F6F8">
<title>学问 · 访问码</title>
<style>
*{box-sizing:border-box}
body{margin:0;min-height:100dvh;display:flex;align-items:center;justify-content:center;padding:20px;
  background:#F5F6F8;color:#191C20;
  font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}
form{width:100%;max-width:330px;background:#fff;border:1px solid #E8EAEE;border-radius:20px;
  padding:26px 22px;box-shadow:0 6px 24px rgba(20,25,40,.06);text-align:center}
h1{font-size:17px;margin:0 0 6px;font-weight:600}
p{font-size:13px;color:#8A9099;margin:0 0 18px;line-height:1.6}
input{width:100%;padding:14px;font-size:22px;letter-spacing:.5em;text-align:center;
  border:1px solid #E8EAEE;border-radius:14px;font-family:inherit;outline:none;background:#fff}
input:focus{border-color:#3355E8}
button{width:100%;margin-top:12px;padding:14px;border:0;border-radius:14px;background:#3355E8;
  color:#fff;font-size:15px;font-family:inherit;cursor:pointer}
.err{color:#D93B3B;font-size:13px;margin:12px 0 0}
</style></head><body>
<form method="post" action="/__gate">
  <h1>输入访问码</h1>
  <input name="code" inputmode="numeric" pattern="[0-9]*" maxlength="4"
         autofocus autocomplete="off" aria-label="访问码">
  <button type="submit">进入</button>
  __ERROR__
</form>
</body></html>"""


def gate_page(error: str = "") -> str:
    return GATE_PAGE.replace("__ERROR__", f'<p class="err">{error}</p>' if error else "")


def _cookies(headers) -> dict:
    out = {}
    for part in (headers.get("cookie") or "").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


class AccessGate:
    """公网入口的访问码闸门。

    挂在 ASGI 层而不是 `@app.middleware("http")`：后者只盖 HTTP，
    `/ws/chat` 会从旁边走过去，等于给隧道留了条不上锁的通道。
    """

    OPEN_PREFIX = ("/static/",)

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)

        from starlette.datastructures import Headers
        headers = Headers(scope=scope)
        if (not needs_code(headers.get("host", ""), headers)
                or cookie_ok(_cookies(headers).get(COOKIE))):
            return await self.app(scope, receive, send)

        path = scope.get("path", "")

        if scope["type"] == "websocket":
            # 未通过访问码：直接关掉，前端会退回 POST（POST 也在闸门后面）
            await send({"type": "websocket.close", "code": 1008})
            return

        if path.startswith(self.OPEN_PREFIX):
            return await self.app(scope, receive, send)

        client = (scope.get("client") or ("?", 0))[0]
        from starlette.responses import FileResponse, HTMLResponse, JSONResponse

        log.info("闸门拦截 %s %s（来自 %s）", scope.get("method", "GET"), path, client)

        if path == "/__gate":
            if scope.get("method", "GET").upper() == "POST":
                # 先把请求体读完，再判断限流：不读会卡住 keep-alive 连接
                body = b""
                while True:
                    msg = await receive()
                    if msg["type"] != "http.request":
                        break
                    body += msg.get("body", b"")
                    if not msg.get("more_body"):
                        break
                if too_many_tries(client):
                    return await HTMLResponse(gate_page("试太多次了，等一分钟再来"),
                                              status_code=429)(scope, receive, send)
                got = (parse_qs(body.decode("utf-8", "replace")).get("code") or [""])[0].strip()
                if got and got == code():
                    # 不做 303 跳转：部分手机 webview 会带着 POST 方法跟 30x，
                    # 落在 GET 路由上就是一坨 {"detail":...} JSON；而且每次跳转
                    # 都要多走一遍公网隧道。直接把学生页和 cookie 一起回掉。
                    log.info("闸门：访问码正确（%s），直接返回主界面", client)
                    resp = FileResponse(WEB_DIR / "index.html",
                                        media_type="text/html; charset=utf-8")
                    resp.set_cookie(COOKIE, cookie_value(), max_age=30 * 86400,
                                    httponly=True, samesite="lax")
                    return await resp(scope, receive, send)
                note_try(client)
                log.warning("闸门：访问码错误（%s）", client)
                return await HTMLResponse(gate_page("访问码不对"), status_code=401)(scope, receive, send)
            return await HTMLResponse(gate_page())(scope, receive, send)

        if path.startswith("/api/"):
            return await JSONResponse({"detail": "需要访问码"}, status_code=401)(scope, receive, send)

        return await HTMLResponse(gate_page(), status_code=401)(scope, receive, send)

