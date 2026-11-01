# -*- coding: utf-8 -*-
"""控制台（Dashboard）专用 API + 学生端心跳。

权限矩阵：
    /api/console/*   控制端专属，非控制端一律 403
    /api/heartbeat   谁都能上报（只是在线状态，不泄露任何数据）
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import DevicePermReq, HeartbeatReq, ModeReq, PurgeReq
from app.config import VERSION
from app.core import access, console_data, llm
from app.core.store import store

router = APIRouter(tags=["console"])


def _peer(request: Request) -> str:
    return access.normalize_peer(request.client.host if request.client else "")


def _client_ip(request: Request) -> str:
    """与 routes_access 同一套规则：走隧道认 X-Forwarded-For 第一个。"""
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return access.normalize_peer(xff.split(",")[0].strip())
    return _peer(request)


def _guard(request: Request) -> None:
    if not access.is_controller(_peer(request), request.headers):
        raise HTTPException(403, "控制台只能在运行服务的电脑上打开")


@router.post("/api/heartbeat")
def heartbeat(req: HeartbeatReq, request: Request):
    """学生页心跳：上报当前会话与进度（在线设备列表的数据源）。

    学生身份就是 IP——服务端按它算 #xxxx 标签，前端不用上报名字。
    顺带把该设备的「生效方案」带回去：控制台改了权限或换了运行方案，
    学生端最多 25 秒就能感知并自刷新，不用重新扫码。
    """
    ip = _client_ip(request)
    console_data.touch(ip, request.headers.get("user-agent", ""),
                       req.sid, req.cur, req.total)
    is_ctrl = access.is_controller(_peer(request), request.headers)
    perm = "server" if is_ctrl else console_data.device_perm(ip)
    return {
        "ok": True,
        "perm": perm,
        "mode": llm.mode() if perm == "server" else "mock",
        "ready": bool(llm.available()) if perm == "server" else False,
        "vision_ready": bool(llm.vision_available()) if perm == "server" else False,
    }


@router.get("/api/console/overview")
def overview(request: Request):
    """总览：运行方案、入口状态、在线设备、最近会话、教学统计——5 秒轮询。"""
    _guard(request)
    port = request.url.port or 8062
    ips = access.lan_ips()
    sessions = console_data.recent_sessions(limit=30)
    acc = access.describe(port)
    return {
        "server": {"version": VERSION},
        "llm": llm.describe(),
        "gate": {
            "code": access.code(),
            "demo": access.demo_mode(),
            "tunnel": access.tunnel.snapshot(),
            "lan_ips": ips,
            "url": f"http://{ips[0]}:{port}" if ips else "",
        },
        # 二维码展示卡的数据源：三种入口的地址（QR 由前端拿 /api/qr 现编）
        # 隧道方案与 token 状态：配置弹层用；token 本身永不下发
        "access": {
            "lan_candidates": acc["lan"]["candidates"],
            "ipv6_url": acc["ipv6"]["url"],
            "tunnel_url": (acc["tunnel"].get("url") or ""
                           if acc["tunnel"].get("state") == "on" else ""),
            "tunnel_state": acc["tunnel"].get("state") or "",
            "tunnel_provider": acc["tunnel"].get("provider") or "cloudflared",
            "tunnel_has_token": bool(acc["tunnel"].get("has_token")),
        },
        "devices": console_data.all_devices(),
        "known_devices": console_data.known(),
        "sessions": sessions,
        "stats": console_data.stats(),
    }


@router.get("/api/console/llm-log")
def llm_log(after: int = 0, request: Request = None):
    """LLM 调用流（增量）。"""
    _guard(request)
    return {"calls": llm.call_log(after), "seq": llm.call_seq()}


@router.get("/api/console/logs")
def logs(after: int = 0, request: Request = None):
    """应用运行日志尾部（增量）。"""
    _guard(request)
    return {"lines": console_data.log_tail(after), "seq": console_data.log_seq()}


@router.get("/api/console/sessions")
def sessions(limit: int = 30, request: Request = None):
    """最近会话（独立端点，供后续会话监控模块用）。"""
    _guard(request)
    limit = max(1, min(100, limit))
    return {"sessions": console_data.recent_sessions(limit=limit)}


@router.get("/api/console/session/{sid}")
def session_detail(sid: str, request: Request = None):
    """单场会话的实时对话流（控制台会话监控弹窗用，前端 2 秒轮询）。"""
    _guard(request)
    d = console_data.session_detail(sid)
    if d is None:
        raise HTTPException(404, "会话不存在或已清库")
    return d


@router.delete("/api/console/session/{sid}")
def delete_session(sid: str, request: Request = None):
    _guard(request)
    try:
        store.delete(sid)
    except Exception as e:
        raise HTTPException(500, f"删除失败：{str(e)[:200]}")
    return {"ok": True}


@router.post("/api/console/sessions/purge")
def purge_sessions(req: PurgeReq, request: Request = None):
    _guard(request)
    if req.scope == "older" and not req.hours:
        raise HTTPException(400, "older 范围必须带 hours")
    n = console_data.purge_sessions(req.scope, req.hours)
    return {"deleted": n}


@router.get("/api/console/devices")
def devices(request: Request = None):
    """近 24h 出现过的设备（设置页的权限管理用，含离线设备）。"""
    _guard(request)
    return {"devices": console_data.all_devices()}


@router.post("/api/console/device-perm")
def device_perm(req: DevicePermReq, request: Request = None):
    """改一台设备的权限：mock（只能用离线题库）/ server（走服务端运行方案）。"""
    _guard(request)
    perm = console_data.set_device_perm(req.ip, req.perm)
    return {"ip": req.ip, "perm": perm}


@router.get("/api/console/perms")
def perms(request: Request = None):
    """所有被授予服务端权限的设备（ID + IP）。

    "配置权限"弹窗的数据源；收回权限复用 POST /device-perm（perm=mock）。
    """
    _guard(request)
    return {"devices": console_data.server_perms()}


@router.post("/api/console/mode")
def switch_mode(req: ModeReq, request: Request = None):
    """快捷切换运行方案（离线/云端/本地/混合），其余配置项原样保留。

    控制台本身只有控制端能进，演示模式不在这里拦——
    演示中恰恰最需要在控制台切方案。
    """
    _guard(request)
    cur = llm.reload()
    llm.save_config(
        base_url=cur["base_url"], model=cur["model"],
        vision_model=cur.get("vision_model"),
        mode=req.mode,
        local_base_url=cur["local_base_url"], local_model=cur["local_model"],
        local_vision_model=cur.get("local_vision_model"),
    )
    return llm.describe()
