# -*- coding: utf-8 -*-
"""测试辅助：强制离线模式、按节点给出"标准答法"、走完一道题。"""
from __future__ import annotations

import os

from app.core import llm
from app.core.engine import Engine, Session


def force_mock() -> None:
    """把模式钉在 mock，避免本机 .env 里的真 Key 让测试走网络。"""
    os.environ["INFERENCE_MODE"] = "mock"
    llm.reload()


def release_mock() -> None:
    os.environ.pop("INFERENCE_MODE", None)
    llm.reload()


def expected_answer(sess: Session) -> str:
    """按引擎当前所处的子状态，给出一个必然判对的说法。"""
    node = sess.nodes[sess.node_idx]
    if node.get("followup") and sess.followup_done:
        return str(node["followup"]["numeric"])
    if node.get("numeric"):
        return str(node["numeric"])
    answers = node.get("answers") or []
    return answers[0] if answers else "不知道"


def walk(sess: Session) -> list[dict]:
    """一路答对，直到题目走完。返回每一步引擎的响应。"""
    steps = []
    for _ in range(30):
        if sess.finished:
            break
        steps.append(Engine.answer(sess, expected_answer(sess)))
    return steps
