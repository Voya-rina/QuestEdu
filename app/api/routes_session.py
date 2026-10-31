# -*- coding: utf-8 -*-
"""学习会话：创建 → 追问 → 自评 → 变式 → 报告。

路由只做参数校验和序列化；节点推进、判定、预警全在 app/core/engine.py。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import AnswerReq, AssessReq, CancelReq, CreateSessionReq
from app.core import access, console_data, llm
from app.core.engine import Engine
from app.core.store import session_lock, store

router = APIRouter(prefix="/api/sessions", tags=["session"])


def _is_controller(request: Request) -> bool:
    peer = request.client.host if request.client else ""
    return access.is_controller(peer, request.headers)


def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return access.normalize_peer(xff.split(",")[0].strip())
    return access.normalize_peer(request.client.host if request.client else "")


@router.post("")
def create_session(req: CreateSessionReq, request: Request):
    # 设备权限：mock 权限的设备只能用离线题库；控制端本机不受限
    if _is_controller(request):
        perm = "server"
    else:
        perm = console_data.device_perm(_client_ip(request))
    cancel = llm.cancel_register(req.req_id) if req.req_id else None
    try:
        sess, payload = Engine.create(
            req.problem.strip(), req.problem_id, req.user.strip(),
            mode="mock" if perm == "mock" else "",
            cancel=cancel,
        )
    except Exception as e:
        raise HTTPException(400, f"拆题失败：{str(e)[:300]}")
    finally:
        if req.req_id:
            llm.cancel_unregister(req.req_id)
    # 学生中途点了"先不做了"：模型就算已经算完也不入库、不返回，
    # 否则会话列表里会躺着一场没人要的会话
    if cancel is not None and cancel.is_set():
        raise HTTPException(400, "拆题已取消")
    store.save(sess)
    return payload


@router.post("/cancel")
def cancel_session(req: CancelReq):
    hit = llm.cancel_trigger(req.req_id)
    return {"ok": True, "hit": hit}


@router.get("/{sid}")
def get_session(sid: str):
    sess = store.get(sid)
    return {
        "session_id": sess.id,
        "problem": sess.problem,
        "problem_id": sess.problem_id,
        "mode": sess.mode,
        "progress": [min(sess.node_idx + 1, len(sess.nodes)), len(sess.nodes)],
        "finished": sess.finished,
        "self_score": sess.self_score,
        "has_variant": bool(sess.variant),
        "created_at": sess.created_at,
    }


@router.post("/{sid}/messages")
def send_message(sid: str, req: AnswerReq):
    if not req.text.strip():
        raise HTTPException(400, "回答不能为空")
    # 同一会话串行：两台设备同时答同一题时不会互相踩状态
    with session_lock(sid):
        sess = store.get(sid)
        res = Engine.answer(sess, req.text.strip())
        store.save(sess)          # 引擎改的是内存对象，这里整体落盘
    return res


@router.post("/{sid}/self-assessment")
def self_assessment(sid: str, req: AssessReq):
    with session_lock(sid):
        sess = store.get(sid)
        res = Engine.self_assess(sess, req.score)
        store.save(sess)
    return res


@router.post("/{sid}/variant")
def make_variant(sid: str):
    with session_lock(sid):
        sess = store.get(sid)
        res = Engine.make_variant(sess)
        store.save(sess)
    return res


@router.post("/{sid}/variant/answer")
def answer_variant(sid: str, req: AnswerReq):
    with session_lock(sid):
        sess = store.get(sid)
        res = Engine.judge_variant(sess, req.text.strip())
        store.save(sess)
    return res


@router.get("/{sid}/report")
def report(sid: str, request: Request):
    data = Engine.report(store.get(sid))
    if not _is_controller(request):
        # 产品铁律是"标准答案只在服务端"：知道会话号的观众设备也不下发
        data["标准答案（服务端留存，未下发）"] = None
    return data
