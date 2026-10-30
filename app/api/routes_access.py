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


@router.get("")
def status(request: Request):
    """三个入口的现状。访问码也在这里下发（能拿到它的本来就是自己人）。"""
    return access.describe(_port(request))


@router.post("/tunnel")
async def tunnel_switch(request: Request):
    """启动 / 停止 Cloudflare 隧道。body: {"action": "start"|"stop"}"""
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
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
