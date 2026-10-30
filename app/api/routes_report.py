# -*- coding: utf-8 -*-
"""报告聚合：雷达图数据与导出。

单独一组路由是因为大纲里它就是这个形状（`/api/reports/{id}/radar`），
跨会话的知识点画像、答辩用的 CSV 也都会挂在这里。
"""
from __future__ import annotations

import csv
import io

from fastapi import APIRouter
from fastapi.responses import Response

from app.core.engine import Engine
from app.core.store import store

router = APIRouter(prefix="/api/reports", tags=["report"])


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
