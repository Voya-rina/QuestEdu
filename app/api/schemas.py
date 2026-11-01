# -*- coding: utf-8 -*-
"""请求体模型（Pydantic）。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class CreateSessionReq(BaseModel):
    problem: str = ""
    problem_id: str | None = None
    # 学生身份就是 IP（服务端算 #xxxx 标签存进 sessions.user）——前端不传名字
    # 前端生成的本次拆题请求令牌：点"先不做了"时拿它来取消，
    # 免得本地模型还在给一个已经没人等的请求卖力
    req_id: str = Field(default="", max_length=64)


class CancelReq(BaseModel):
    """取消一次进行中的拆题请求（对应 CreateSessionReq.req_id）。"""
    req_id: str = Field(min_length=4, max_length=64)


class AnswerReq(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class AssessReq(BaseModel):
    # 五分制（0~5）。以前是 le=100 的百分制遗留：发 88 会被引擎静默钳成 5 分
    score: int = Field(ge=0, le=5)


class HeartbeatReq(BaseModel):
    """学生页心跳：在线设备列表的数据源。字段全可空——能带多少带多少。

    学生身份就是 IP（服务端算 #xxxx 标签），前端不用上报名字。
    """
    sid: str = Field(default="", max_length=32)
    cur: int | None = None
    total: int | None = None


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


class ModeReq(BaseModel):
    """控制台快捷切换运行方案（只动 INFERENCE_MODE，其余配置原样保留）。"""
    mode: str = Field(pattern="^(mock|cloud|local|hybrid)$")


class DevicePermReq(BaseModel):
    """设备权限：mock=只能用离线题库；server=走服务端当前运行方案。"""
    ip: str = Field(min_length=1, max_length=64)
    perm: str = Field(pattern="^(mock|server)$")


class PurgeReq(BaseModel):
    """会话批量清理：all=全部；finished=非进行中；older=hours 小时前。"""
    scope: str = Field(pattern="^(all|finished|older)$")
    hours: float | None = Field(default=None, gt=0, le=24 * 30)
