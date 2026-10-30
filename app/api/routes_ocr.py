# -*- coding: utf-8 -*-
"""拍照识别入口。"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, File, UploadFile

from app.core import ocr

router = APIRouter(prefix="/api", tags=["ocr"])


@router.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    """上传题目照片，返回 LaTeX。

    识别失败也返回 200 + `ok:false` + 一句人话——前端据此引导学生手动输入，
    而不是弹一个红色报错框。

    识别（本地看图模型可能要跑几分钟）必须丢进线程池：async 处理器里直接
    跑同步调用会把事件循环整个堵住，其他设备的请求全部冻结——
    和 /ws/chat 之前同款问题。
    """
    data = await file.read()
    return await asyncio.to_thread(ocr.extract, data, file.content_type)
