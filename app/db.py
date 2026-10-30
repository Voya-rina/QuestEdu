# -*- coding: utf-8 -*-
"""数据库连接。

默认 SQLite（`data/questedu.db`），换 PostgreSQL 只需改 `DB_URL`——
这也是大纲里"单机跑得住、上云换得掉"的那句话的落点。

    DB_URL=postgresql+psycopg://user:pass@host/db
"""
from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import DATA_DIR


def _default_url() -> str:
    return f"sqlite:///{(DATA_DIR / 'questedu.db').as_posix()}"


DB_URL = os.environ.get("DB_URL") or _default_url()
_IS_SQLITE = DB_URL.startswith("sqlite")

engine = create_engine(
    DB_URL,
    future=True,
    # SQLite 默认禁止跨线程复用连接，FastAPI 的线程池会踩到
    connect_args={"check_same_thread": False} if _IS_SQLITE else {},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    from app import models  # noqa: F401  建表需要模型先被导入
    Base.metadata.create_all(engine)
