# -*- coding: utf-8 -*-
"""学习会话：创建 → 追问 → 自评 → 变式 → 报告。

路由只做参数校验和序列化；节点推进、判定、预警全在 app/core/engine.py。
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas import AnswerReq, AssessReq, CreateSessionReq
from app.core.engine import Engine
from app.core.store import session_lock, store

router = APIRouter(prefix="/api/sessions", tags=["session"])


@router.post("")
def create_session(req: CreateSessionReq):
    try:
        sess, payload = Engine.create(req.problem.strip(), req.problem_id, req.user.strip())
    except Exception as e:
        raise HTTPException(400, f"拆题失败：{str(e)[:300]}")
    store.save(sess)
    return payload


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
def report(sid: str):
    return Engine.report(store.get(sid))
