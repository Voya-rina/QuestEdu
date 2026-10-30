# -*- coding: utf-8 -*-
"""报告聚合：雷达图数据与导出。

单独一组路由是因为大纲里它就是这个形状（`/api/reports/{id}/radar`），
跨会话的知识点画像、答辩用的 CSV 也都会挂在这里。
"""
from __future__ import annotations

import csv
import io

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from app.core import access
from app.core.engine import Engine
from app.core.store import store

router = APIRouter(prefix="/api/reports", tags=["report"])


def _is_controller(request: Request) -> bool:
    peer = request.client.host if request.client else ""
    return access.is_controller(peer, request.headers)


@router.get("/history")
def history(limit: int = 30, user: str = "", request: Request = None):
    """首页「学习记录」：最近会话的轻量列表。

    故意不走 store.get()——那条路会把每条消息都拉出来，列表页用不上。
    直接查 sessions 表，汇总字段从 JSON 列里现场算。
    传了 user 就只看这个人的：记录按用户分开，不堆在一起。

    权限：非控制端必须带 user——观众设备不该有能力翻所有人的记录；
    控制端（演示者的电脑）不带 user 时可以看到全部，便于排查。
    """
    from sqlalchemy import select

    from app.db import SessionLocal
    from app.models import SessionRow

    if request is not None and not user.strip() and not _is_controller(request):
        raise HTTPException(403, "学习记录按用户查询，请带上 user 参数")

    limit = max(1, min(100, limit))
    stmt = select(SessionRow).order_by(SessionRow.created_at.desc()).limit(limit)
    if user.strip():
        stmt = stmt.where(SessionRow.user == user.strip()[:64])
    with SessionLocal() as s:
        rows = s.scalars(stmt).all()

    out = []
    for r in rows:
        results = r.results or []
        done = [x for x in results if x.get("verdict") in ("correct", "revealed")]
        mistakes = sum(
            1 for x in results
            if x.get("verdict") in ("wrong", "revealed") or x.get("wrong", 0) > 0
        )
        out.append({
            "id": r.id,
            "subject": r.subject or r.problem_id or "自定义题",
            "problem": r.problem,
            "mode": r.mode,
            "status": r.status,                     # running / finished / assessed
            "self_score": r.self_score,
            "created_at": r.created_at.isoformat(timespec="seconds"),
            "nodes_done": len(done),
            # plan 列在老库行上可能是 NULL，直接 .get() 会 500 掉整个记录列表
            "nodes_total": len((r.plan or {}).get("nodes") or []),
            "mistakes": mistakes,
        })
    return {"sessions": out}


@router.get("/{sid}/radar")
def radar(sid: str):
    return Engine.radar(store.get(sid))


@router.get("/{sid}/export.csv")
def export_csv(sid: str):
    """导出这次会话的全部数据。

    带 UTF-8 BOM：Excel 双击打开中文不乱码（这是答辩现场唯一会被用到的格式）。
    """
    sess = store.get(sid)
    report = Engine.report(sess)
    buf = io.StringIO()
    w = csv.writer(buf)

    w.writerow(["QuestEdu 会话导出"])
    w.writerow(["会话", sess.id])
    w.writerow(["题目", sess.problem])
    w.writerow(["模式", sess.mode])
    w.writerow(["开始时间", sess.created_at])
    w.writerow([])

    w.writerow(["知识点掌握"])
    w.writerow(["知识点", "掌握度", "说明"])
    for bar in report["bars"]:
        w.writerow([bar["kp"], bar["score"], bar["note"]])
    w.writerow([])

    w.writerow(["作答记录"])
    w.writerow(["节点", "知识点", "判定", "答错次数"])
    for res in sess.results:
        w.writerow([res["n"], res["knowledge"], res["verdict"], res["wrong"]])
    w.writerow([])

    w.writerow(["元认知校准"])
    w.writerow(["自评分", "预警", "依据"])
    w.writerow([
        sess.self_score if sess.self_score is not None else "",
        (sess.warning or {}).get("title", ""),
        (sess.warning or {}).get("reason", ""),
    ])
    if sess.variant_result:
        w.writerow([])
        w.writerow(["变式题"])
        w.writerow(["题目", sess.variant["problem"]])
        w.writerow(["学生答案", sess.variant_result["student"]])
        w.writerow(["标准答案", sess.variant_result["answer"]])
        w.writerow(["结果", "正确" if sess.variant_result["correct"] else "错误"])

    body = "\ufeff" + buf.getvalue()
    return Response(
        content=body.encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="questedu-{sid}.csv"'},
    )
