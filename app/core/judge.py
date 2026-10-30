# -*- coding: utf-8 -*-
"""答案判定层（大纲「答案判定策略」一节的落地）。

原则：**能算的算，算不了的对表述，最后才谈对错；宁可澄清，不可误判。**

三层，越靠前越硬：
    1. 符号计算层  节点有数值答案 → SymPy 直接比对（"2" "２" "2/4" "0.5"）
    2. 表述归一化  全角 / 中文数字 / 上标 / 空格先统一，再匹配一组可接受说法
    3. 语义兜底    只有开了云端、且前两层都不中，才交 LLM 判语义

学生随口一答"二阶""x²""第二阶"，不该被判错——这是这一层存在的全部理由。
"""
from __future__ import annotations

import re

from app.core import llm, mathcheck

# 学生讨要答案的典型说法（大纲「防绕过设计」）
BYPASS_PHRASES = [
    "给我答案", "告诉我答案", "直接说答案", "直接告诉我", "别问了",
    "你直接说", "答案呢", "直接给", "说答案",
]

_FULL2HALF = str.maketrans("０１２３４５６７８９（）　", "0123456789() ")
_CN_DIGIT = {"零": "0", "一": "1", "二": "2", "两": "2", "三": "3",
             "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}


def normalize(text: str) -> str:
    """把学生随口写的东西归一化。

    统一后 "二阶 / 2阶 / 第二阶 / ２阶 / x² / X^2" 才会落到同一个判定上。
    """
    t = (text or "").translate(_FULL2HALF).lower()
    t = (t.replace("²", "^2").replace("³", "^3").replace("⁴", "^4")
          .replace("⁵", "^5").replace("⁶", "^6"))
    t = re.sub(r"\s+", "", t)
    for k, v in _CN_DIGIT.items():
        t = t.replace(k, v)
    return t


def hits(text: str, words) -> bool:
    """归一化后做包含匹配。words 为空时恒为 False。"""
    t = normalize(text)
    return any(normalize(w) and normalize(w) in t for w in (words or []))


def hint_for(node: dict, level: int) -> str:
    """取第 level 级提示（0 起）。云端生成的节点可能没有 hints，兜底回原问题。"""
    hs = node.get("hints") or []
    if not hs:
        return node.get("question", "再想一想？")
    if level < 0:
        return hs[-1]
    return hs[min(level, len(hs) - 1)]


def verdict_offline(text: str, node: dict):
    """离线判定。返回 (verdict, got, reply_override)。

    reply_override 非空表示"这句回复已经写好了"——留给那些思路本身值得回应的答案，
    比如学生用等价无穷小：方向没错但这题会失效，得当场讲清楚为什么，
    而不是丢一句"答错了"。
    """
    num = node.get("numeric")
    if num:
        want = mathcheck.parse_expr(num)
        got = mathcheck.number_in(text)
        # want 解析不出来（模型给的是"计算 y'=3x^2-3"这种带说明的串）就当没有：
        # 不能因为一条不可求值的期望值，把答对的学生判成错
        if got is not None and want is not None:
            return ("correct" if mathcheck.same(got, want) else "wrong"), got, ""
        # 没抽到数字，可能是"二阶""x²"这类表述，继续往下走

    if hits(text, node.get("answers")):
        return "correct", None, ""
    for branch in node.get("branches", []):
        if hits(text, branch["kw"]):
            return branch["verdict"], None, branch["reply"]
    if hits(text, node.get("partial_kw")):
        return "partial", None, ""
    return "wrong", None, ""


CLOUD_JUDGE_SYSTEM = """你是数学助教，只做一件事：判定学生的这一句回答，在"当前这一小步"上是否成立。

严格的 JSON 输出，不要 Markdown：
{"verdict":"correct|partial|wrong","extracted_value":"学生给出的最终数值，没有就空字符串","stage":"回顾|启发|纠偏|收束","reply":"你对学生说的话"}

extracted_value 的要求（这一格会被程序直接喂给 SymPy 复核）：
- 只写数值本身：1/2、0.5、-16、0、sqrt(2)/2、3*x^2-3
- 纯 ASCII，不要单位、不要汉字、不要在式子里带等号、不要 $ 和 LaTeX
- 学生没给出具体数值（只是说思路、说方法）时写空字符串 ""，不要替他补

硬规则（最高优先级）：
1. 绝不给出最终答案，绝不给完整解题过程。
2. 只针对当前这一步回应，一次只问一个问题。
3. 学生答错时，用更小的问题把他拉回到能答对的地方；不要直接讲正确答案。
4. 学生说"直接给我答案/别问了"时，共情一句，然后把问题拆得更小。
5. **reply 必须以一个问句结尾**：学生看完必须清楚下一步该回答什么。提示和问题写在同一条消息里，
   不要先发一条纯陈述、等学生随便回一句再补问题。
6. reply 控制在 60 字以内，可用 $...$ 写 LaTeX。"""


def verdict_cloud(text: str, node: dict, history: list):
    """在线判定：LLM 判语义，SymPy 覆盖数值结论。

    第 3 层（语义兜底）只在云端生效——离线模式下前两层仍管用，
    这是"断网也能演示"的基础。
    """
    ctx = "\n".join(f"{h['role']}: {h['text']}" for h in history[-6:])
    expect = node.get("expect") or "、".join(node.get("answers") or [])
    user = (
        f"当前这一小步的问法：「{node.get('question', '')}」\n"
        f"期望学生答出的要点：「{expect}」\n"
        f"近期对话：\n{ctx}\n\n学生这次说：{text}"
    )
    data = llm.chat_json(CLOUD_JUDGE_SYSTEM, user)
    verdict = str(data.get("verdict", "wrong")).lower()
    if verdict not in ("correct", "partial", "wrong"):
        verdict = "wrong"

    extracted = str(data.get("extracted_value", "") or "")
    if extracted and node.get("numeric"):
        got = mathcheck.number_in(extracted)
        want = mathcheck.parse_expr(node["numeric"])
        if got is not None and want is not None:
            # 符号引擎说话，模型让位——护城河就落在这三行
            verdict = "correct" if mathcheck.same(got, want) else "wrong"

    return verdict, data.get("reply", ""), data.get("stage", node.get("stage", "纠偏"))
