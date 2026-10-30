# -*- coding: utf-8 -*-
"""LLM 配置：查询 / 保存 / 连通性自检。

Key 只写服务端 .env，永不下发前端（只回前 6 位做提示）。
"""
from __future__ import annotations

from fastapi import APIRouter

from app.api.schemas import LLMConfigReq
from app.core import llm

router = APIRouter(prefix="/api/llm", tags=["llm"])


@router.get("")
def status():
    return llm.describe()


@router.post("/config")
def save(req: LLMConfigReq):
    """写入 .env 并立即生效。

    `mode=mock` 就是"离线题库"这个运行方案本身，并不是清除；
    `api_key` 留空表示沿用已保存的 Key，真删要显式传 `clear_key`。
    """
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
def test():
    return llm.test()


@router.get("/models")
def models():
    """当前端点上有哪些模型。模型没起时返回空列表，不报错。"""
    return {"models": llm.list_models()}
