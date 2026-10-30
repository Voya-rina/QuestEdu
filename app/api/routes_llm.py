# -*- coding: utf-8 -*-
"""LLM 配置：查询 / 保存 / 连通性自检。

Key 只写服务端 .env，永不下发前端（只回前 6 位做提示）。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import LLMConfigReq
from app.core import access, llm

router = APIRouter(prefix="/api/llm", tags=["llm"])


def _is_controller(request: Request) -> bool:
    peer = request.client.host if request.client else ""
    return access.is_controller(peer, request.headers)


def _ctrl_guard(request: Request) -> None:
    """改模型配置是控制端专属。观众设备发来的 POST 一律 403，
    光靠前端藏齿轮按钮不算数。"""
    if not _is_controller(request):
        raise HTTPException(403, "模型设置只能在运行服务的电脑上修改")


def _demo_guard() -> None:
    """演示模式中，设置相关写操作一律拒绝（前端灰掉不算数，POST 能绕）。"""
    if access.demo_mode():
        raise HTTPException(403, "演示模式已开启，设置已锁定；只能在开启演示的设备上解锁")


@router.get("")
def status(request: Request):
    d = llm.describe()
    controller = _is_controller(request)
    d["controller"] = controller
    d["demo_locked"] = access.demo_mode()
    # controller=False 时前端把整个设置入口藏掉；unlockable 决定控制端上
    # 显不显示 🔒 按钮（演示中观众设备连按钮都拿不到）
    xff = request.headers.get("x-forwarded-for", "")
    ip = xff.split(",")[0].strip() if xff else (request.client.host if request.client else "")
    d["demo_unlockable"] = controller or ip == access.demo_ip()
    return d


@router.post("/config")
def save(req: LLMConfigReq, request: Request):
    """写入 .env 并立即生效。

    `mode=mock` 就是"离线题库"这个运行方案本身，并不是清除；
    `api_key` 留空表示沿用已保存的 Key，真删要显式传 `clear_key`。
    """
    _ctrl_guard(request)
    _demo_guard()
    llm.save_config(
        req.base_url, req.api_key, req.model,
        vision_model=req.vision_model,
        mode=req.mode,
        local_base_url=req.local_base_url,
        local_model=req.local_model,
        local_vision_model=req.local_vision_model,
        clear_key=req.clear_key,
    )
    return llm.describe()


@router.post("/test")
def test(request: Request):
    _ctrl_guard(request)
    _demo_guard()
    return llm.test()


@router.get("/models")
def models():
    """当前端点上有哪些模型。模型没起时返回空列表，不报错。"""
    return {"models": llm.list_models()}
