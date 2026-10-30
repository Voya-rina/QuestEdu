# -*- coding: utf-8 -*-
"""流式追问通道 `/ws/chat`（大纲 API 草案里的 WS 项）。

协议：
    客户端 → {"session_id": "...", "text": "学生说的话"}
    服务端 → {"type":"delta","text":"…"}     分片
             {"type":"done", ...}            与 POST /messages 完全相同的载荷
             {"type":"error","message":"…"}  会话失效等

分片目前来自引擎已经算好的整段回复——教学状态机必须一次算完才知道
"这一步结束了没有"。等接上真模型的 token 流，只需把 `_chunks` 换成
上游的 SSE 增量，前端一行不用改。
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from app.core.engine import Engine
from app.core.store import store

router = APIRouter(tags=["stream"])

CHUNK = 14          # 每个分片的字符数：手机上看起来像在打字
DELAY = 0.018       # 分片间隔，太快了看不出流式，太慢了像卡住


def _chunks(text: str) -> list[str]:
    return [text[i:i + CHUNK] for i in range(0, len(text), CHUNK)] or [""]


@router.websocket("/ws/chat")
async def chat_stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_json()
            sid = (msg.get("session_id") or "").strip()
            text = (msg.get("text") or "").strip()
            if not text:
                await ws.send_json({"type": "error", "message": "回答不能为空"})
                continue
            try:
                sess = store.get(sid)
            except HTTPException as e:
                await ws.send_json({"type": "error", "message": e.detail})
                continue

            res = Engine.answer(sess, text)
            store.save(sess)

            for piece in _chunks(res["message"]):
                await ws.send_json({"type": "delta", "text": piece})
                await asyncio.sleep(DELAY)
            await ws.send_json({"type": "done", **res})
    except WebSocketDisconnect:
        return
    except Exception as e:                       # 兜底：别让异常把连接挂死
        try:
            await ws.send_json({"type": "error", "message": str(e)[:200]})
        except Exception:
            pass
