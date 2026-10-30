# -*- coding: utf-8 -*-
"""访问入口：局域网地址 / 隧道开关 / 托管链接 / 访问码。

三个入口是并列的，不是互斥的：局域网给同一 WiFi 用，隧道给校园网隔离时用，
托管给"电脑关了也想看"用。开关只控制"开不开"，不控制"用哪个"。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.core import access

router = APIRouter(prefix="/api/access", tags=["access"])


def _port(request: Request) -> int:
    return request.url.port or 80


def _client_ip(request: Request) -> str:
    """发起请求的客户端 IP。走隧道时认 X-Forwarded-For 的第一个。
    双栈监听下 IPv4 对端带 ::ffff: 前缀，统一还原后再比较。"""
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return access.normalize_peer(xff.split(",")[0].strip())
    return access.normalize_peer(request.client.host if request.client else "")


def _can_unlock(request: Request) -> bool:
    """这台设备能不能解锁演示模式：只认开启者那台机器的 IP。

    不能用 is_local_host——它的白名单含全部局域网地址，观众手机
    也在同一 WiFi 下，等于没锁。
    """
    return (not access.demo_mode()) or _client_ip(request) == access.demo_ip()


def _controller(request: Request) -> bool:
    """是不是控制端（运行服务的电脑）。注意认 socket 对端，不认转发头。"""
    peer = request.client.host if request.client else ""
    return access.is_controller(peer, request.headers)


def _ctrl_guard(request: Request) -> None:
    """入口管理（隧道/托管/改码/演示开关）是控制端专属，用户端一律 403。"""
    if not _controller(request):
        raise HTTPException(403, "这些设置只能在运行服务的电脑上操作")


@router.get("")
def status(request: Request):
    """三个入口的现状。访问码只下发给控制端（观众看到的是打码）。"""
    return access.describe(_port(request), demo_unlockable=_can_unlock(request),
                           controller=_controller(request))


@router.post("/tunnel")
async def tunnel_switch(request: Request):
    """启动 / 停止 Cloudflare 隧道。body: {"action": "start"|"stop"}"""
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    _ctrl_guard(request)
    action = str((body or {}).get("action") or "start").lower()

    if action == "stop":
        return access.tunnel.stop()
    try:
        return access.tunnel.start(_port(request))
    except access.AccessError as e:
        raise HTTPException(400, str(e))


@router.post("/host")
async def host_link(request: Request):
    """记下 WorkBuddy 托管的分享链接（只用于展示，发布/下线在对话里做）。"""
    _ctrl_guard(request)
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    try:
        url = access.set_host_url(str((body or {}).get("url") or ""))
    except access.AccessError as e:
        raise HTTPException(400, str(e))
    return {"url": url}


@router.post("/code")
async def set_code(request: Request):
    """改访问码。改完旧的 cookie 自动失效（token 跟着 code 走）。"""
    _ctrl_guard(request)
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    try:
        code = access.set_code(str((body or {}).get("code") or ""))
    except access.AccessError as e:
        raise HTTPException(400, str(e))
    return {"code": code, "token": access.cookie_value()}


@router.post("/demo")
async def demo_switch(request: Request):
    """演示模式开关。body: {"on": bool}

    开：谁都能开（把设置锁上没有损失），并记下开启设备的 IP。
    关：只有开启的那台机器（或演示者本机）能关——观众设备上
        这个按钮根本不显示，就算直接发 POST 也会被 IP 校验拦下。
    """
    _ctrl_guard(request)
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    on = bool((body or {}).get("on"))
    if not on and not _can_unlock(request):
        raise HTTPException(403, "只有开启演示模式的那台设备可以解锁")
    access.set_demo(on, ip=_client_ip(request))
    return {"on": access.demo_mode(), "unlockable": _can_unlock(request)}
