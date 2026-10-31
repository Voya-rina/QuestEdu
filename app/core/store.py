# -*- coding: utf-8 -*-
"""会话存储。

两个实现，接口一致（save / get / count / all / exists）：

    MemoryStore   进程内存，重启即清空（测试与兜底）
    SqlStore      SQLite 五表，服务重启后会话仍在

选择：默认 `SqlStore`；建库失败（比如没装驱动）时退回内存并打印警告，
保证演示现场不会因为数据库问题开天窗。强制内存：`STORE=memory`。
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, timedelta

from fastapi import HTTPException
from sqlalchemy import delete, select

from app.core.engine import Engine, Session


# ---------------------------------------------------------------- 会话级锁
#
# FastAPI 的同步路由跑在线程池里：不同设备的不同会话天然并行，互不干扰。
# 但同一个会话被两台设备同时提交时（比如共用一个二维码），引擎改的是
# store 缓存里的同一个 Session 对象——加一把按 sid 分的锁，串行化同会话
# 的写操作，跨会话照常并行。
_sess_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()

# 进行中的会话这么久没有任何新消息（作答/回复都算），就自动关闭——
# 学生关了页面忘掉的会话不该永远挂在"进行中"里
STALE_SESSION_MINUTES = 30


def session_lock(sid: str) -> threading.Lock:
    with _locks_guard:
        return _sess_locks.setdefault(sid, threading.Lock())


class MemoryStore:
    def __init__(self) -> None:
        self._data: dict[str, Session] = {}
        self._deleted: set[str] = set()

    def close_stale(self, minutes: float = STALE_SESSION_MINUTES) -> int:
        """超时关闭：进行中且最近 messages 时间早于 cutoff 的会话标记 closed。"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        n = 0
        for s in self._data.values():
            if s.closed or s.finished or s.self_score is not None:
                continue
            last = s.history[-1]["at"] if s.history else s.created_at
            try:
                if datetime.fromisoformat(str(last)) < cutoff:
                    s.closed = True
                    n += 1
            except Exception:
                continue
        return n

    def save(self, sess: Session) -> None:
        # 删除后在途请求晚到落盘会把会话整体写回——墓碑拦掉，删除才算删干净
        if sess.id in self._deleted:
            self._deleted.discard(sess.id)
            return
        self._data[sess.id] = sess

    def get(self, sid: str) -> Session:
        sess = self._data.get(sid)
        if sess is None:
            raise HTTPException(404, "会话不存在")
        return sess

    def exists(self, sid: str) -> bool:
        return sid in self._data

    def count(self) -> int:
        return len(self._data)

    def all(self) -> list[Session]:
        return list(self._data.values())

    def clear(self) -> None:
        self._data.clear()

    def delete(self, sid: str) -> None:
        self._data.pop(sid, None)
        self._deleted.add(sid)

    def ids(self) -> list[tuple[str, str, object]]:
        """(id, status, created_at) 轻量列表，批量清理用。"""
        out = []
        for s in self._data.values():
            status = ("assessed" if s.self_score is not None
                      else "finished" if s.finished else "running")
            out.append((s.id, status, s.created_at))
        return out


class SqlStore:
    """写穿缓存：内存里留着活对象给引擎改，每次改完整体落盘。"""

    def __init__(self) -> None:
        from app import db
        from app.models import (
            MasteryRow, MessageRow, SelfAssessmentRow, SessionRow, StepRow,
        )

        self._db = db
        self._SessionRow = SessionRow
        self._MessageRow = MessageRow
        self._StepRow = StepRow
        self._SelfRow = SelfAssessmentRow
        self._MasteryRow = MasteryRow
        self._cache: dict[str, Session] = {}
        self._deleted: set[str] = set()
        db.init_db()

    # ---------------------------------------------------------- 写

    def save(self, sess: Session) -> None:
        if sess.id in self._deleted:
            self._deleted.discard(sess.id)
            return
        self._cache[sess.id] = sess
        db = self._db
        with db.SessionLocal() as s:
            row = s.get(self._SessionRow, sess.id)
            if row is None:
                row = self._SessionRow(id=sess.id)
                s.add(row)

            row.created_at = _parse_dt(sess.created_at)
            row.user = (getattr(sess, "user", "") or "")[:64]
            row.subject = sess.plan.get("题型") or ""
            row.problem = sess.problem
            row.problem_id = sess.problem_id
            row.mode = sess.mode
            # 超时关闭优先：一旦判 closed，后续落盘不能再把它写回 running
            row.status = ("closed" if sess.closed
                          else "assessed" if sess.self_score is not None
                          else "finished" if sess.finished else "running")
            row.plan = sess.plan
            row.results = sess.results
            row.node_idx = sess.node_idx
            row.node_wrong = sess.node_wrong
            row.followup_done = sess.followup_done
            row.prereq_active = sess.prereq_active
            row.prereq_used = sess.prereq_used
            row.self_score = sess.self_score
            row.warning = sess.warning
            row.variant = sess.variant
            row.variant_result = sess.variant_result

            self._sync_messages(s, sess)
            self._sync_steps(s, sess)
            self._sync_self_assessment(s, sess)
            self._sync_mastery(s, sess)
            s.commit()

    def _sync_messages(self, db_session, sess: Session) -> None:
        db_session.execute(delete(self._MessageRow).where(self._MessageRow.session_id == sess.id))
        for m in sess.history:
            db_session.add(self._MessageRow(
                session_id=sess.id,
                role=m.get("role", ""),
                content=m.get("text", ""),
                stage=m.get("stage"),
                node_no=m.get("node"),
                kind=m.get("kind"),
                created_at=_parse_dt(m.get("at")),
            ))

    def _sync_steps(self, db_session, sess: Session) -> None:
        db_session.execute(delete(self._StepRow).where(self._StepRow.session_id == sess.id))
        seen = set()
        for node in sess.nodes:
            if node["n"] in seen:
                continue
            seen.add(node["n"])
            db_session.add(self._StepRow(
                session_id=sess.id,
                n=node["n"],
                knowledge=node.get("knowledge", ""),
                question=node.get("question", ""),
                stage=node.get("stage", ""),
                expect=node.get("expect", "") or "、".join(node.get("answers") or []),
                sympy_check=str(node.get("numeric") or ""),
            ))

    def _sync_self_assessment(self, db_session, sess: Session) -> None:
        db_session.execute(delete(self._SelfRow).where(self._SelfRow.session_id == sess.id))
        if sess.self_score is None:
            return
        warned = bool(sess.warning and sess.warning.get("level") == "red")
        actual = None
        gap = None
        if sess.variant_result:
            actual = 5 if sess.variant_result["correct"] else 0
            gap = sess.self_score - actual
        db_session.add(self._SelfRow(
            session_id=sess.id,
            step_id=None,
            score=sess.self_score,
            actual=actual,
            gap=gap,
            warned=warned,
        ))

    def _sync_mastery(self, db_session, sess: Session) -> None:
        db_session.execute(delete(self._MasteryRow).where(self._MasteryRow.session_id == sess.id))
        for bar in Engine.report(sess)["bars"]:
            if bar["note"] == "未涉及":
                continue
            total, correct = 0, 0
            for node in sess.nodes:
                if node.get("knowledge") != bar["kp"]:
                    continue
                res = next((r for r in sess.results if r["n"] == node["n"]), None)
                if res is None:
                    continue
                total += 1
                if res["verdict"] == "correct" and res["wrong"] == 0:
                    correct += 1
            db_session.add(self._MasteryRow(
                session_id=sess.id,
                knowledge=bar["kp"],
                correct=correct,
                total=total,
                score=bar["score"],
            ))

    # ---------------------------------------------------------- 读

    def get(self, sid: str) -> Session:
        if sid in self._cache:
            return self._cache[sid]
        with self._db.SessionLocal() as s:
            row = s.get(self._SessionRow, sid)
            if row is None:
                raise HTTPException(404, "会话不存在")
            sess = self._to_session(s, row)
        self._cache[sid] = sess
        return sess

    def _to_session(self, db_session, row) -> Session:
        history = [
            {"role": m.role, "text": m.content, "at": m.created_at.isoformat(timespec="seconds"),
             **({"stage": m.stage} if m.stage else {}),
             **({"node": m.node_no} if m.node_no is not None else {}),
             **({"kind": m.kind} if m.kind else {})}
            for m in db_session.scalars(
                select(self._MessageRow).where(self._MessageRow.session_id == row.id)
                .order_by(self._MessageRow.id))
        ]
        return Session(
            id=row.id,
            problem=row.problem,
            mode=row.mode,
            plan=row.plan or {"nodes": []},
            problem_id=row.problem_id or "",
            user=getattr(row, "user", "") or "",
            node_idx=row.node_idx,
            node_wrong=row.node_wrong,
            followup_done=row.followup_done,
            prereq_active=row.prereq_active,
            prereq_used=row.prereq_used,
            results=list(row.results or []),
            history=history,
            self_score=row.self_score,
            warning=row.warning,
            variant=row.variant,
            variant_result=row.variant_result,
            closed=(row.status == "closed"),
            created_at=row.created_at.isoformat(timespec="seconds"),
        )

    def exists(self, sid: str) -> bool:
        if sid in self._cache:
            return True
        with self._db.SessionLocal() as s:
            return s.get(self._SessionRow, sid) is not None

    def close_stale(self, minutes: float = STALE_SESSION_MINUTES) -> int:
        """超时关闭：status=running 且最后一条消息早于 cutoff 的会话标记 closed。

        「无响应」按最后一条消息（作答/回复都算）的时间算，而不是会话创建时间。
        """
        cutoff = datetime.now() - timedelta(minutes=minutes)
        victims: list[str] = []
        with self._db.SessionLocal() as s:
            running = s.scalars(
                select(self._SessionRow.id).where(self._SessionRow.status == "running")
            ).all()
            for sid in running:
                last = s.scalars(
                    select(self._MessageRow.created_at)
                    .where(self._MessageRow.session_id == sid)
                    .order_by(self._MessageRow.created_at.desc()).limit(1)
                ).first()
                if last is None:
                    last = s.get(self._SessionRow, sid).created_at
                if last is not None and last.replace(tzinfo=None) < cutoff:
                    victims.append(sid)
            for sid in victims:
                s.get(self._SessionRow, sid).status = "closed"
            if victims:
                s.commit()
        for sid in victims:      # 缓存里的活对象同步标记，后续 save 不会写回 running
            csess = self._cache.get(sid)
            if csess is not None:
                csess.closed = True
        return len(victims)

    def count(self) -> int:
        with self._db.SessionLocal() as s:
            return len(s.scalars(select(self._SessionRow.id)).all())

    def all(self) -> list[Session]:
        with self._db.SessionLocal() as s:
            ids = list(s.scalars(select(self._SessionRow.id).order_by(
                self._SessionRow.created_at.desc())))
        return [self.get(i) for i in ids]

    def clear(self) -> None:
        """只清缓存，不动数据库——用来模拟"服务重启"。"""
        self._cache.clear()

    def delete(self, sid: str) -> None:
        self._cache.pop(sid, None)
        self._deleted.add(sid)
        with self._db.SessionLocal() as s:
            for table in (self._MessageRow, self._StepRow,
                          self._SelfRow, self._MasteryRow):
                s.execute(delete(table).where(table.session_id == sid))
            s.execute(delete(self._SessionRow).where(self._SessionRow.id == sid))
            s.commit()

    def ids(self) -> list[tuple[str, str, object]]:
        with self._db.SessionLocal() as s:
            rows = s.execute(
                select(self._SessionRow.id, self._SessionRow.status,
                       self._SessionRow.created_at)
            ).all()
        return [(r[0], r[1], r[2]) for r in rows]


def _parse_dt(value) -> datetime:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return datetime.now()


def _make_store():
    if os.environ.get("STORE", "").lower() == "memory":
        return MemoryStore()
    try:
        return SqlStore()
    except Exception as e:  # 驱动缺失 / 目录不可写：退回内存，保证能演示
        print(f"[store] SQLite 初始化失败，退回内存模式：{e}")
        return MemoryStore()


store = _make_store()
