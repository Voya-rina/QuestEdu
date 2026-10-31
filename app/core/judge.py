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


def _expr_answer_ok(text: str, want) -> bool:
    """答案本身是表达式（导函数、切线方程这类）时的比对。

    以前这种节点只能靠关键词整串匹配——期望值是 "y'=3x^2-3"，学生老老实实
    打 "3x^2-3" 反而判错。现在复用变式题的机制：学生输入整体解析成 SymPy
    表达式（容忍 4x^3、y=9x-16、f(x)=x^2），再符号比对。
    """
    got = mathcheck.parse_student_expr(text)
    return got is not None and mathcheck.same(got, want)


def verdict_offline(text: str, node: dict):
    """离线判定。返回 (verdict, got, reply_override)。

    reply_override 非空表示"这句回复已经写好了"——留给那些思路本身值得回应的答案，
    比如学生用等价无穷小：方向没错但这题会失效，得当场讲清楚为什么，
    而不是丢一句"答错了"。
    """
    num = node.get("numeric")
    if num:
        want = mathcheck.parse_expr(num)
        # want 解析不出来（模型给的是"计算 y'=3x^2-3"这种带说明的串）就当没有：
        # 不能因为一条不可求值的期望值，把答对的学生判成错。
        if want is not None:
            if want.is_number:
                got = mathcheck.number_in(text)
                if got is not None:
                    return ("correct" if mathcheck.same(got, want) else "wrong"), got, ""
            else:
                # 期望值是表达式（3*x**2-3、9*x-16）：整体解析学生输入做符号比对
                if _expr_answer_ok(text, want):
                    return "correct", None, ""
        # 没抽到数字/解析不出，可能是"二阶""x²"这类表述，继续往下走

    if hits(text, node.get("answers")):
        return "correct", None, ""
    for branch in node.get("branches", []):
        if hits(text, branch["kw"]):
            return branch["verdict"], None, branch["reply"]
    if hits(text, node.get("partial_kw")):
        return "partial", None, ""
    return "wrong", None, ""


CLOUD_JUDGE_SYSTEM = """你是数学解题程序里的判定器。学生正按小步解题，你只判定他刚提交的一句回答，并回一句话。

输入分四段：
【当前问题】学生要回答的那一小步
【期望要点】这一步该答出的内容
【近期对话】上下文
【学生这次说】判定对象（只判这一句，不要替他补答）

第一步，判 verdict（三选一）：
- correct：回答给出了期望要点的具体内容（数值 / 式子 / 名词）
- partial：方向对但不完整，只答了一半
- wrong：答非所问、看不懂题、拒绝作答，或只是敷衍话
判定从宽：答案是期望要点的数值一致形式、简写或省略说法，都算 correct
（例：期望「分子分母都是 0」，学生答「0」= correct）

敷衍话规则（最容易判错的地方）：
"对""好""嗯""继续""？""好的""是的""明白了"这类话不是数学回答。
- 当前问题要求具体数学内容时，这类回答一律判 wrong
- 只有当前问题本身是是非题（例："这一步等于 1 吗？"）时，"对 / 不对"才算回答
- 判 wrong 时绝不顺着敷衍话往下走

第二步，给 extracted_value：学生回答里的最终数值，纯 ASCII（如 1/2、0.5、-16、sqrt(2)/2、3*x^2-3）。
没有数值就写空串 ""，不要替他编。

第三步，写 reply（对学生说的话，≤60 字，必须以问号结尾）：
- **reply 一律用简体中文**，哪怕学生用英文或其他语言作答
- **问句必须是具体的数学问题**——能让学生写出一个数字、一个式子或一个名词的那种。
  禁止"这样理解了吗""明白了吗""下一步我们该做什么""还有其他问题吗"这类空话
- correct：只写简短肯定（如"对，就是这样。"），**不要再提任何问题**——
  下一个问题由程序自动接上，你多问一句就会和它打架
  （实测会蹦出"请确认一下导数的定义域？"这种没头没脑的追问）；
  **判了 correct 就绝不能再质疑答案、也绝不要再要求补充说明这一步**
- wrong：绝不给答案、不复述原话、不用"是不是……"的句式把答案提示出来；
  把当前问题拆成更小更直白的一问
  （例：问展开阶数不会，就先问"x 的几次方项是决定性的？"）
- partial：指出缺了哪一半，再问他补上
- **只许围绕【当前问题】提问**：学生答错时把这一步拆小，绝不许跳回之前
  已经答对/讲过的步骤重新问一遍
- 针对学生这次说了什么来回应，不要自顾自推进流程；LaTeX 用 $...$

只输出 JSON，不要 Markdown，不要解释：
{"verdict":"correct|partial|wrong","extracted_value":"","reply":""}"""


def verdict_cloud(text: str, node: dict, history: list):
    """在线判定：LLM 判语义，SymPy 覆盖数值结论。

    第 3 层（语义兜底）只在云端生效——离线模式下前两层仍管用，
    这是"断网也能演示"的基础。
    阶段标签不信任模型：它爱把 stage 当创作发挥，导致界面上
    回顾/启发/纠偏乱跳。stage 一律由代码按当前节点指定。
    """
    ctx = "\n".join(f"{h['role']}: {h['text']}" for h in history[-6:])
    expect = node.get("expect") or "、".join(node.get("answers") or [])
    user = (
        f"【当前问题】{node.get('question', '')}\n"
        f"【期望要点】{expect}\n"
        f"【近期对话】\n{ctx}\n\n"
        f"【学生这次说】{text}"
    )
    data = llm.chat_json(CLOUD_JUDGE_SYSTEM, user)
    verdict = str(data.get("verdict", "wrong")).lower()
    if verdict not in ("correct", "partial", "wrong"):
        verdict = "wrong"

    # 数值核对是最后一道闸，模型说了不算：
    # 优先用模型抠出的值，抠不出来再从学生原话里抽一次（小模型经常漏抽）
    # 纯数值期望 → 抽数字比对；表达式期望（导函数、切线方程）→ 整体解析比对。
    # 表达式若只靠语义判定，模型判"correct"无从校验，判错也发现不了
    if node.get("numeric"):
        want = mathcheck.parse_expr(node["numeric"])
        if want is not None and want.is_number:
            got = None
            for src in (str(data.get("extracted_value", "") or ""), text):
                if src:
                    got = mathcheck.number_in(src)
                    if got is not None:
                        break
            if got is not None:
                # 符号引擎说话，模型让位——护城河就落在这几行
                verdict = "correct" if mathcheck.same(got, want) else "wrong"
        elif want is not None and getattr(want, "free_symbols", set()):
            if _expr_answer_ok(text, want):
                verdict = "correct"

    return verdict, data.get("reply", ""), node.get("stage")
