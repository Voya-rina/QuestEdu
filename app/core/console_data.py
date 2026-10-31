# -*- coding: utf-8 -*-
"""控制台（Dashboard）数据源：设备心跳表、应用日志环形缓冲、最近会话查询。

- 心跳：学生页每 25 秒 POST /api/heartbeat，超过 ONLINE_WINDOW 秒没心跳即离线
- 环形日志：内存里留最近 500 条格式化日志供增量拉取（落盘的仍是 data/questedu.log）
- 会话：直接查 sessions 表最近 N 条（复用学习记录的轻量查询思路）
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime

from app.config import DATA_DIR

# ---------------------------------------------------------------- 设备心跳

_lock = threading.Lock()
_devices: dict[str, dict] = {}
ONLINE_WINDOW = 90          # 秒：超过没心跳即视为离线

_PERMS: dict[str, str] = {}
_PERM_FILE = DATA_DIR / "device_perms.json"
_PERM_LOCK = threading.Lock()

try:
    if _PERM_FILE.exists():
        _PERMS = {k: "server" for k in json.loads(_PERM_FILE.read_text(encoding="utf-8"))
                  if isinstance(k, str)}
except Exception:
    _PERMS = {}


def device_perm(ip: str) -> str:
    """设备权限：mock（只能用离线题库）或 server（走服务端当前运行方案）。"""
    with _PERM_LOCK:
        return _PERMS.get(ip, "mock")


def set_device_perm(ip: str, perm: str) -> str:
    with _PERM_LOCK:
        if perm == "server":
            _PERMS[ip] = "server"
        else:
            _PERMS.pop(ip, None)
        try:
            _PERM_FILE.write_text(json.dumps(sorted(_PERMS)), encoding="utf-8")
        except Exception:
            pass
        return _PERMS.get(ip, "mock")

_KIND_TOKENS = ("mobile", "android", "iphone", "ipad", "harmony")


def _kind(ua: str) -> str:
    ua = (ua or "").lower()
    return "手机" if any(k in ua for k in _KIND_TOKENS) else "电脑"


def touch(ip: str, user: str = "", ua: str = "", sid: str = "",
          cur=None, total=None) -> None:
    """记录/刷新一台设备的心跳。

    每次**新连接**（90 秒内没见过）都把该设备权限重置回 MOCK：
    授权只在设备持续在线期间有效，重连一律从离线题库重新开始——
    控制台要给谁开服务端权限，就趁它在线时再点一次。
    """
    now = time.time()
    with _lock:
        prev = _devices.get(ip) or {}
        fresh = (not prev) or (now - prev.get("seen", 0) > ONLINE_WINDOW)
        _devices[ip] = {
            "ip": ip,
            "user": (user or "").strip()[:64] or prev.get("user", ""),
            "kind": _kind(ua),
            "sid": (sid or "").strip() or prev.get("sid", ""),
            "cur": cur if isinstance(cur, int) else prev.get("cur"),
            "total": total if isinstance(total, int) else prev.get("total"),
            "seen": now,
            "first": prev.get("first", now),
        }
    if fresh:
        _reset_perm(ip)


def _reset_perm(ip: str) -> None:
    with _PERM_LOCK:
        if _PERMS.pop(ip, None) is not None:
            try:
                _PERM_FILE.write_text(json.dumps(sorted(_PERMS)), encoding="utf-8")
            except Exception:
                pass


def _row(d: dict, now: float, online: bool) -> dict:
    out = dict(d)
    out["perm"] = device_perm(d["ip"])
    out["online"] = online
    out["ago"] = max(0, int(now - d["seen"]))
    return out


def presence_rows() -> list[dict]:
    """在线设备列表（新的在前）。能出现在这里就是在线的。"""
    now = time.time()
    with _lock:
        rows = [d for d in _devices.values() if now - d["seen"] <= ONLINE_WINDOW]
    rows.sort(key=lambda d: -d["seen"])
    return [_row(d, now, True) for d in rows]


def all_devices() -> list[dict]:
    """近 24h 出现过的设备（设置页的权限管理用，含离线）。"""
    now = time.time()
    cutoff = now - 24 * 3600
    with _lock:
        rows = [d for d in _devices.values() if d["seen"] >= cutoff]
    rows.sort(key=lambda d: -d["seen"])
    return [_row(d, now, now - d["seen"] <= ONLINE_WINDOW) for d in rows]


def known(hours: float = 24) -> int:
    """最近 hours 小时里出现过的设备数。"""
    cutoff = time.time() - hours * 3600
    with _lock:
        return sum(1 for d in _devices.values() if d["seen"] >= cutoff)


# ---------------------------------------------------------------- 日志环形缓冲

class RingLog(logging.Handler):
    """内存环形日志：控制台增量拉取用。"""

    def __init__(self, capacity: int = 500):
        super().__init__()
        self._dq: deque = deque(maxlen=capacity)
        self._seq = 0
        self._lk = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
        except Exception:
            return
        with self._lk:
            self._seq += 1
            self._dq.append({"seq": self._seq, "line": line})

    def tail(self, after: int = 0) -> list[dict]:
        with self._lk:
            return [dict(e) for e in self._dq if e["seq"] > after]

    def cur_seq(self) -> int:
        with self._lk:
            return self._seq


ring_log = RingLog()


def log_tail(after: int = 0) -> list[dict]:
    return ring_log.tail(after)


def log_seq() -> int:
    return ring_log.cur_seq()


# ---------------------------------------------------------------- 最近会话

def recent_sessions(limit: int = 30) -> list[dict]:
    """最近会话的轻量列表（与学习记录同一套字段思路，供控制台监控）。

    控制台每 5 秒轮一次这里——顺手把超时的会话关掉，
    学生忘了关的"进行中"不会永远挂着。
    """
    from sqlalchemy import select

    from app.core.store import store as _store
    from app.db import SessionLocal
    from app.models import SessionRow

    _store.close_stale()
    with SessionLocal() as s:
        rows = s.scalars(
            select(SessionRow).order_by(SessionRow.created_at.desc()).limit(limit)
        ).all()

    out = []
    for r in rows:
        plan = r.plan or {}
        total = len(plan.get("nodes") or [])
        out.append({
            "id": r.id,
            "user": (getattr(r, "user", "") or ""),
            "subject": r.subject or "",
            "problem": (r.problem or "")[:60],
            "mode": r.mode,
            "status": r.status,                       # running / finished / assessed
            "progress": [min(r.node_idx + 1, total), total],
            "node_wrong": r.node_wrong,
            "self_score": r.self_score,
            "created": r.created_at.isoformat(timespec="seconds"),
        })
    if out and rows:
        now_ts = datetime.now().timestamp()
        for item, r in zip(out, rows):
            created = r.created_at.timestamp() if r.created_at else now_ts
            item["age_s"] = max(0, int(now_ts - created))
    return out


def session_detail(sid: str) -> dict | None:
    """单场会话的实时监控数据：对话流 + 进度 + 自评/变式结果。

    控制台点开某场会话时用，前端每 2 秒轮一次——学生答一步，
    控制端当场能看到。取不到（已清库/内存态重启）返回 None。
    """
    from app.core.store import store

    try:
        sess = store.get(sid)
    except Exception:
        return None
    hist = [
        {
            "role": m.get("role", "system"),
            "text": m.get("text", ""),
            "at": m.get("at", ""),
            "kind": m.get("kind", ""),
            "stage": m.get("stage", ""),
        }
        for m in list(getattr(sess, "history", []))[-80:]      # 控制端看尾部即可
    ]
    variant = sess.variant or {}
    vres = sess.variant_result or {}
    total = len(sess.nodes) if sess.nodes else 0
    return {
        "id": sid,
        "user": (getattr(sess, "user", "") or ""),
        "problem": sess.problem,
        "subject": sess.plan.get("题型") or "",
        "mode": sess.mode,
        "status": ("closed" if getattr(sess, "closed", False)
                   else "assessed" if sess.self_score is not None
                   else "finished" if sess.finished else "running"),
        "progress": [min(sess.node_idx + 1, total), total] if total else [0, 0],
        "self_score": sess.self_score,
        "variant_problem": variant.get("problem", ""),
        "variant_ok": vres.get("correct"),
        "messages": hist,
    }


def purge_sessions(scope: str, hours: float | None = None) -> int:
    """批量删除会话（连库里的消息/步骤/自评/掌握度一起删）。

    scope: all=全部；finished=所有非进行中；older=创建时间早于 hours 小时前。
    返回删除场数。
    """
    from datetime import datetime, timedelta

    from app.core.store import store

    cutoff = (datetime.now() - timedelta(hours=hours)) if hours is not None else None

    def _as_dt(v):
        if isinstance(v, datetime):
            return v
        try:
            return datetime.fromisoformat(str(v))
        except Exception:
            return None

    victims = []
    for sid, status, created in store.ids():
        if scope == "all":
            victims.append(sid)
        elif scope == "finished":
            if status != "running":
                victims.append(sid)
        elif scope == "older":
            dt = _as_dt(created)
            if cutoff is not None and dt is not None and dt < cutoff:
                victims.append(sid)
    for sid in victims:
        store.delete(sid)
    return len(victims)


def stats(limit: int = 100) -> dict:
    """教学概览：近 limit 场会话的量与质。

    「校准一致」＝做了变式验证的场次里，SymPy 判定学生真的会做的比例——
    这是产品核心主张（被讲懂 ≠ 会做）的直接度量。「高频卡点」按节点
    累计答错次数排，答题设计/讲题重点往这看。
    """
    from datetime import datetime, time as _time

    from sqlalchemy import select

    from app.db import SessionLocal
    from app.models import SessionRow

    today0 = datetime.combine(datetime.now().date(), _time.min)
    with SessionLocal() as s:
        rows = s.scalars(
            select(SessionRow).order_by(SessionRow.created_at.desc()).limit(limit)
        ).all()

    total = len(rows)
    today = sum(1 for r in rows if r.created_at and r.created_at >= today0)
    running = sum(1 for r in rows if r.status == "running")
    # 超时关闭的场次不算进"完答率"的分母完成项——它是被放弃的，不是被完成的
    closed = sum(1 for r in rows if r.status == "closed")
    done = total - running - closed
    with_variant = [r for r in rows
                    if isinstance(r.variant_result, dict)
                    and isinstance(r.variant_result.get("correct"), bool)]
    calib_ok = sum(1 for r in with_variant if r.variant_result["correct"])
    hotspot: dict[str, int] = {}
    for r in rows:
        for res in (r.results or []):
            if isinstance(res, dict) and res.get("wrong"):
                k = res.get("knowledge") or "未知"
                hotspot[k] = hotspot.get(k, 0) + 1
    top = sorted(hotspot.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    return {
        "total": total,
        "today": today,
        "running": running,
        "done": done,
        "finish_rate": round(done * 100 / total) if total else 0,
        "calib": {"ok": calib_ok, "n": len(with_variant)},
        "hotspots": [{"knowledge": k, "wrong": v} for k, v in top],
    }
