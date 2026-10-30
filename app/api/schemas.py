# -*- coding: utf-8 -*-
"""请求体模型（Pydantic）。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class CreateSessionReq(BaseModel):
    problem: str = ""
    problem_id: str | None = None


class AnswerReq(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class AssessReq(BaseModel):
    score: int = Field(ge=0, le=100)


class LLMConfigReq(BaseModel):
    """模型配置。

    mode 为空表示"按有没有 Key 自动判断"，兼容老的调用方式。
    看图模型单独两个字段：云端一个、本地一个。本地文本模型（qwen3.8 之类）
    不吃图片，必须显式指定多模态模型才允许走拍照。
    """
    mode: str = ""
    base_url: str = ""
    # 留空（None）＝沿用已保存的 Key。密码框每次打开都是空的，
    # 把空值当"清空"会让用户改个模型名就把 Key 弄丢。
    api_key: str | None = None
    clear_key: bool = False
    model: str = ""
    vision_model: str | None = None
    local_base_url: str = ""
    local_model: str = ""
    local_vision_model: str | None = None
