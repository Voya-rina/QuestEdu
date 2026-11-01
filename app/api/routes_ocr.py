# -*- coding: utf-8 -*-
"""拍照识别入口。"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, File, Request, UploadFile

from app.core import access, console_data, llm, ocr

router = APIRouter(prefix="/api", tags=["ocr"])


def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return access.normalize_peer(xff.split(",")[0].strip())
    return access.normalize_peer(request.client.host if request.client else "")


@router.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...), request: Request = None):
    """上传题目照片，返回 LaTeX。

    识别失败也返回 200 + `ok:false` + 一句人话——前端据此引导学生手动输入，
    而不是弹一个红色报错框。

    识别（本地看图模型可能要跑几分钟）必须丢进线程池：async 处理器里直接
    跑同步调用会把事件循环整个堵住，其他设备的请求全部冻结——
    和 /ws/chat 之前同款问题。

    调用上下文（用户指纹）在丢线程池之前设置：to_thread 会复制上下文，
    OCR 的 LLM 调用日志才能对上人。
    """
    data = await file.read()
    llm.set_call_context(user=console_data.label_for_ip(_client_ip(request)))
    return await asyncio.to_thread(ocr.extract, data, file.content_type)
