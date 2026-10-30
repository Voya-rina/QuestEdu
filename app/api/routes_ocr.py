# -*- coding: utf-8 -*-
"""拍照识别入口。"""
from __future__ import annotations

from fastapi import APIRouter, File, UploadFile

from app.core import ocr

router = APIRouter(prefix="/api", tags=["ocr"])


@router.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    """上传题目照片，返回 LaTeX。

    识别失败也返回 200 + `ok:false` + 一句人话——前端据此引导学生手动输入，
    而不是弹一个红色报错框。
    """
    data = await file.read()
    return ocr.extract(data, file.content_type)
