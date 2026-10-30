# -*- coding: utf-8 -*-
"""五张表，照大纲「数据存储」一节落地。

    sessions          会话：题目、模式、进度、live 状态（plan/results 用 JSON 存）
    messages          对话流水：谁说的、说了什么、当时在哪个阶段
    steps             解题节点快照：问法、期望答案、SymPy 校验表达式
    self_assessments  自评记录：分数、实际对错、偏差、是否预警
    mastery           知识点掌握：答对次数 / 总次数 / 掌握度

雷达图 = mastery 按知识点聚合；虚假自信记录 = self_assessments 里"高分答错"的子集。

会话的 live 状态（plan、results、variant…）用 JSON 列存在 sessions 里：
它们结构随引擎演进，拆成关系表会让每次改字段都要迁移，得不偿失。
分析用的维度（messages/steps/mastery）才值得建表。
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def _now() -> datetime:
    return datetime.now()


class SessionRow(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, index=True)
    user: Mapped[str] = mapped_column(String(64), default="")         # 归属用户（学习记录按人分）
    subject: Mapped[str] = mapped_column(String(64), default="")      # 题型，如"极限计算"
    problem: Mapped[str] = mapped_column(Text, default="")            # 题目原文（LaTeX）
    problem_id: Mapped[str] = mapped_column(String(32), default="")
    mode: Mapped[str] = mapped_column(String(16), default="mock")
    status: Mapped[str] = mapped_column(String(16), default="running")  # running/finished/assessed

    # ---- live 状态（引擎直接读写，JSON 存）----
    plan: Mapped[dict] = mapped_column(JSON, default=dict)
    results: Mapped[list] = mapped_column(JSON, default=list)
    node_idx: Mapped[int] = mapped_column(Integer, default=0)
    node_wrong: Mapped[int] = mapped_column(Integer, default=0)
    followup_done: Mapped[bool] = mapped_column(Boolean, default=False)
    prereq_active: Mapped[bool] = mapped_column(Boolean, default=False)
    prereq_used: Mapped[bool] = mapped_column(Boolean, default=False)
    self_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    warning: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    variant: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    variant_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class MessageRow(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(16), ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(16))          # student / assistant / system
    content: Mapped[str] = mapped_column(Text, default="")
    stage: Mapped[str | None] = mapped_column(String(16), nullable=True)
    node_no: Mapped[int | None] = mapped_column(Integer, nullable=True)
    kind: Mapped[str | None] = mapped_column(String(16), nullable=True)   # variant / warning
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class StepRow(Base):
    __tablename__ = "steps"
    __table_args__ = (UniqueConstraint("session_id", "n", name="uq_step_session_n"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(16), ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    n: Mapped[int] = mapped_column(Integer)
    knowledge: Mapped[str] = mapped_column(String(64), default="")
    question: Mapped[str] = mapped_column(Text, default="")
    stage: Mapped[str] = mapped_column(String(16), default="")
    expect: Mapped[str] = mapped_column(Text, default="")
    sympy_check: Mapped[str] = mapped_column(String(128), default="")   # 期望值表达式，如 1/2


class SelfAssessmentRow(Base):
    __tablename__ = "self_assessments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(16), ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    step_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    score: Mapped[int] = mapped_column(Integer, default=0)
    actual: Mapped[int | None] = mapped_column(Integer, nullable=True)      # 100 / 0，由变式结果裁决
    gap: Mapped[float | None] = mapped_column(Float, nullable=True)         # 自评 - 实际
    warned: Mapped[bool] = mapped_column(Boolean, default=False)            # 是否亮红灯
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class MasteryRow(Base):
    __tablename__ = "mastery"
    __table_args__ = (UniqueConstraint("session_id", "knowledge", name="uq_mastery_session_kp"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(16), ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    knowledge: Mapped[str] = mapped_column(String(64), default="")
    correct: Mapped[int] = mapped_column(Integer, default=0)   # 一次答对的节点数
    total: Mapped[int] = mapped_column(Integer, default=0)     # 涉及的节点数
    score: Mapped[int] = mapped_column(Integer, default=0)     # 掌握度 0~100
