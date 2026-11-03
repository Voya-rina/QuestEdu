# -*- coding: utf-8 -*-
"""教学引擎：苏格拉底式追问 + 自评校准。

设计原则（对应大纲「Prompt 护栏」一节）：
  1. 流程由代码状态机控制 —— 推不推进节点、第几次答错给哪级提示、
     什么时候弹自评，全是 Python 的 if/else，不指望 LLM 自觉。
  2. 答案由 SymPy 决定 —— 学生报出的数值先过符号引擎，LLM 的判定可以被覆盖。
  3. 标准答案只在服务端 —— 下发给前端的永远只有"下一步问什么"。
"""
from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache

from app.core import llm, mathcheck
from app.core.judge import BYPASS_PHRASES, hint_for, hits, verdict_cloud, verdict_offline
from app.data.problems import PROBLEMS, get as get_problem

log = logging.getLogger(__name__)

MAX_WRONG_PER_NODE = 3


@lru_cache(maxsize=32)
def _truth_of(problem: str):
    """按题面缓存的符号真值（隐函数求导才有）。

    拆题和判分各要查一次，重算一次几十毫秒；判定是每个答案都走的路径，
    不值得每次重算。
    """
    return mathcheck.implicit_derivative(problem)


# 判定学生答案时能用的符号闸，顺序即优先级
_NODE_CHECKS = (
    mathcheck.equation_diff_ok,
    mathcheck.dy_expr_ok,
    mathcheck.term_deriv_ok,
    mathcheck.x_term_deriv_ok,
)


def symbolic_accepts(node: dict, text: str, truth) -> bool:
    """学生的回答在符号层算不算对（只认这一步期望的那一类）。

    期望是完整等式时，学生只给一个单项导数不能算过。对不上就不表态，
    判定权交回上一层——宁可漏判，不可误判。
    """
    if not truth or not (text or "").strip():
        return False
    exp = node.get("expect") or ""
    for chk in _NODE_CHECKS:
        try:
            if chk(exp, truth) is True and chk(text, truth) is True:
                return True
        except Exception:
            continue
    return False



def choices_of(node: dict | None) -> list:
    """节点可选的选择题选项；没有就是空列表，前端据此决定要不要画选项条。"""
    return list((node or {}).get("choices") or [])


# 判对之后不该再出现的话：让学生重做/重说/确认，或者又抛一个问题
_REAFFIRM_NG = re.compile(
    r"请|写出|写一|说明|解释|确认|重新|告诉我|想一想|试一|再算|再想|验证|对吗|是不是")


def _affirm(reply: str, node: dict, fallback: str = "") -> str:
    """判对之后开口说的那一句。

    判定模型偶尔判了 correct 却回一句"请写出原方程的具体形式"——自相矛盾，
    学生会以为自己的答案不算数。这类跑偏的回复换掉，换成节点自带的肯定语
    或一句通用的；短肯定原样留着。
    """
    t = (reply or "").strip()
    if t and len(t) <= 40 and "？" not in t and "?" not in t and not _REAFFIRM_NG.search(t):
        return t
    return fallback or node.get("correct_reply") or "对，就是这样。"


def ensure_question(text: str) -> str:
    """最后一道保险：保证每条回复都以问句收尾。

    模型的回复经常以陈述句结尾（"…应用洛必达法则。"），以前这里会补一句
    "那你觉得呢？"——正是实测里学生最烦的那种无意义尾巴。现在改成把陈述
    句转成真正的疑问句（"…法则，结果是多少？"），本就是问句的原样放行。
    """
    t = (text or "").strip()
    if not t:
        return "再想一想？"
    return _to_question(t)


# 模型偶尔会把"教学意图"写成教师视角的备注（"让学生掌握求导规则"），
# 那是给老师看的，学生对着它没法回答
_TEACHER_TALK = re.compile(
    r"^(这(一)?步(要|需要)?(让|使|引导|要求)|让|使|引导|要求)(学生|用户)?(去|来)?"
)


def clean_ask(text: str) -> str:
    """尽量把教师视角的备注改成人话。改不干净时由 ensure_question 兜底。"""
    return _TEACHER_TALK.sub("", (text or "").strip())


# 题目自带的降维问句常以"先退一步/先退一小步"开头，而引擎拼装时已经
# 加了"**先退一步**："——不剥掉就成了"先退一步：先退一小步：…"复读机
_RETREAT_PREFIX = re.compile(
    r"^[^。？?！!\n]{0,12}?(先退一小步|先退一步|退一小步|退一步|先热个身|先热热身)[：:，,]\s*"
)


def strip_retreat_prefix(text: str) -> str:
    """剥掉前置小问句开头那层"退一步"式的铺垫，避免和引擎的引导语重复。"""
    return _RETREAT_PREFIX.sub("", (text or "").strip())


# ---------------------------------------------------------------- 数据结构

@dataclass
class Session:
    id: str
    problem: str
    mode: str
    plan: dict
    problem_id: str = "limit"
    user: str = ""                    # 归属用户；空串兼容老数据
    node_idx: int = 0
    node_wrong: int = 0
    followup_done: bool = False       # 当前节点的"追问"是否已经抛出
    prereq_active: bool = False       # 正在问前置知识点（降维中）
    prereq_used: bool = False         # 本次节点是否已经降维过一次
    results: list = field(default_factory=list)      # 每节点判定记录
    history: list = field(default_factory=list)
    self_score: int | None = None
    warning: dict | None = None
    variant: dict | None = None
    variant_result: dict | None = None
    closed: bool = False             # 超时自动关闭（30 分钟无任何作答/回复）
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def nodes(self) -> list:
        return self.plan["nodes"]

    @property
    def finished(self) -> bool:
        return self.node_idx >= len(self.nodes)

    def log(self, role: str, text: str, **extra):
        self.history.append({
            "role": role, "text": text,
            "at": datetime.now().isoformat(timespec="seconds"), **extra,
        })


# ---------------------------------------------------------------- 计划生成

def demo_plan(prob: dict) -> dict:
    """内置题库的计划。答案全部由 SymPy 现算，题库里不写死数字。"""
    answer = mathcheck.evaluate(prob["答案表达式"])
    # 内置题库的 numeric 也过一遍清洗：不变量只有一条——
    # 进到节点里的 numeric 要么是空串，要么一定能被 parse_expr 解析
    nodes = []
    for nd in prob["nodes"]:
        node = dict(nd)
        if node.get("numeric"):
            node["numeric"] = mathcheck.clean_numeric(node["numeric"])
        nodes.append(node)
    return {
        "题型": prob["题型"],
        "知识点": prob["知识点"],
        "难度": prob["难度"],
        # 学生永远看不到这三个字段
        "_answer_latex": mathcheck.latex(answer) if answer is not None else "?",
        "_answer_plain": mathcheck.plain(answer) if answer is not None else "?",
        "_answer_note": prob.get("答案备注", ""),
        "nodes": nodes,
    }


PLAN_SYSTEM = """你是数学助教，把一道题拆成学生逐步作答的小问题。只输出 JSON，不要解释：
{"题型":"","知识点":[""],"难度":"","节点":[{"问":"","答":"","式":""}]}

规则简版（细节照示例学）：
- 「问」：一步只问一件事的具体疑问句（…是多少/几/什么？），记号要学生打得出；
  一个节点只有一个问号，禁止把两步塞进同一句；
  禁止流程反问（"然后呢""明白了吗"）、指令式（"写出…"）、把答案喂到嘴边、节点间重复
- 「答」：一口气说得完的一个数 / 式子 / 名词；「式」是它的 SymPy 形式（纯 ASCII，
  如 1/2、sqrt(2)/2、E-E**3），非数值写 ""
  「答」禁止写成等式或整条方程（移项后的等式也不行）——只写一侧的式子或一个数
- 节点 = 标准解法的必要步骤：先在心里完整解一遍，按解题顺序原样拆开；
  禁止对解题没帮助的代数变形（如隐函数求导里合并对数——逐项直接求导才是正路）
- **最后一个节点必须问出题目最终要求的那个量，它答完题目就结束**；
  求出最终答案之后禁止再追加任何节点（方法总结、思路回顾、变形探讨都不要）

示例一（∫[0,1] x·e^x dx）：
{"题型":"定积分计算","知识点":["分部积分法"],"难度":"中等","节点":[
{"问":"被积函数是两种不同类型函数的乘积，这种积分用哪个方法？","答":"分部积分","式":""},
{"问":"令 u=x，那么 v=∫e^x dx 等于多少？","答":"e^x","式":"exp(x)"},
{"问":"代回分部积分公式，原函数是 x*e^x-e^x。代入上下限，结果是几？","答":"1","式":"1"}]}

示例二（xy²−ln(x+1)+ln y=1，y=y(x)，求 x=0 处的 dy/dx）——这是标准拆法，
照这个样子和顺序拆：**逐项直接对 x 求导（含 y 的项求导后尾巴带 y'）→ 解出 y' →
代数值 → 求出 dy/dx 即到最后一节点，后面没有任何节点**；全程不做对数合并、
通分之类的方程变形。第一个节点就直接算数学（代值或求某项的导数），
禁止拿"ln y 可化为 ln x + ln y""能不能把对数合并/化简"这类问题开场——
ln y 不存在"拆成 ln x + ln y"这种变形；对数只有 ln a + ln b = ln(ab) 一条
真实法则，且在本类题里禁用：
{"题型":"隐函数求导","知识点":["链式法则","乘积法则"],"难度":"中等","节点":[
{"问":"xy² 这个乘积对 x 的导数是谁？","答":"y²+2xyy'","式":""},
{"问":"方程两边同时关于自变量 x 求导（y 是 y(x)）后，等式变形为什么？","答":"y²+2xyy'-1/(x+1)+y'/y=0","式":""},
{"问":"把含 y' 的项归到一边解出 y'，y' 等于多少（用 x 和 y 表示）？","答":"y'=(1/(x+1)−y²)/(2xy+1/y)","式":""},
{"问":"现在需要具体数值了。把 x=0 代回原方程，y 等于几？","答":"e","式":"E"},
{"问":"把 x=0、y=e 代进 y' 的表达式，dy/dx 等于几？","答":"e-e³","式":"E-E**3"}]}"""

_STAGES = ["回顾", "启发", "纠偏", "收束"]

# 结构坏了也别整段丢：模型漏 key 时（实测 Gemma 4 写出
# {"n":1,"提问":"...","分部积分的变量选择","e^x"}），问句本身还是好的
_FIELD_ASK = re.compile(r'"(?:提问|问|教学意图)"\s*:\s*"((?:[^"\\]|\\.)*)"')
_FIELD_ANS = re.compile(r'"(?:期望答案|答)"\s*:\s*"((?:[^"\\]|\\.)*)"')
_FIELD_EXP = re.compile(r'"sympy检查"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _unescape(s: str) -> str:
    """把正则抠出来的字符串还原成普通文本（主要是 \\\\ 和 \\" 两种情况）。"""
    return s.replace('\\\\"', '"').replace("\\\\", "\\")


def _plan_budget() -> int:
    """节点数上限已取消：拆题步数交给模型按标准解法定，不再按模式设上限。"""
    return 10**9


# 流程性反问 / 开场白：学生没法回答，出现即说明节点是废的。
# 这是实测里小模型最常犯的毛病——第一个"问"就写成"那你觉得呢？"，
# 引擎的 ensure_question 只会往后面再补一个问号，越补越糊涂
_META_Q = re.compile(
    r"你想|你觉得|你认不认为|要不要|好了吗|准备好了|可以开始|我们先|先从(哪个|哪里)"
    r"|继续(吗|吧|么)|好不好|行不行|然后呢|怎么办|明白了吗|听懂了吗|接下来"
)

# 判定说错、话术却在肯定（实测：符号答案 e-e³ 被数值闸抽残翻成 wrong，
# reply 却是"对，就是这样"）——学生被夸完再被重问同一句，永远走不到终点。
# 以判定为准：肯定话术丢弃，走标准纠错路径。
_AFFIRM_REPLY = re.compile(
    r"对[的，,。！!]|就是这样|没错|完全正确|回答正确|答对了|正是|正确[！!。]"
)


def _real_question(q: str) -> bool:
    """一个节点问句是否"能答"：以问号结尾、不是流程反问、不是教师备注。"""
    t = (q or "").strip()
    if not t.endswith(("?", "？")):
        return False
    if _META_Q.search(t):
        return False
    if _TEACHER_TALK.match(t):
        return False
    if _BOGUS_LOG.search(t):
        return False
    return True


# 实测模型会编造对数"恒等式"当考察点（"lny 可化为 lnx+lny？"——数学上
# 根本不成立），还爱拿它当第一个节点。对数词 + 拆/合动词同现即废节点。
_BOGUS_LOG = re.compile(
    r"(?:ln|对数)[^。？！?!]{0,24}(?:可化为|化为|拆成|拆为|合并为|合并成|写成)"
    r"|(?:可化为|拆成|拆为|合并为|合并成|写成)[^。？！?!]{0,24}(?:ln|对数)",
    re.I,
)


# 模型爱写指令句（"将 x=2 代入…计算 y 值。"），ensure_question 只会补
# "那你觉得呢？"，越补越糊涂。这里把陈述句改写成真正的疑问句。
_STATEMENT_END = re.compile(r"[。；;.\s]+$")
_FIND_VERB = re.compile(r"计算|代入|求出?|写出|化简|展开|解出|得到|算出")


def _to_question(text: str) -> str:
    """把节点问句尽量修成可回答的疑问句；已经是问句就原样返回。"""
    t = (text or "").strip()
    if t.endswith(("?", "？")):
        return t
    t = _STATEMENT_END.sub("", t).rstrip()
    if not t:
        return t
    # 取最后一小句来补问（"…。它们的和是5，积是6。" → "…积是6，是多少？"）
    if _FIND_VERB.search(t):
        return t + "，结果是多少？"
    return t + "，是多少？"


def extract_questions(raw: str) -> list[dict]:
    """从一段坏掉的 JSON 里，按字段名捞出节点。

    这不优雅，但比"整道题失败"好：字段名对得上，教学流程就能继续，
    判定层大不了退化成关键词匹配。
    """
    asks = _FIELD_ASK.findall(raw)
    if not asks:
        return []
    ans = _FIELD_ANS.findall(raw)
    exp = _FIELD_EXP.findall(raw)
    out = []
    for i, a in enumerate(asks):
        out.append({
            "提问": _unescape(a),
            "期望答案": _unescape(ans[i]) if i < len(ans) else "",
            "sympy检查": _unescape(exp[i]) if i < len(exp) else "",
            "知识点": "",
        })
    return out


# 只写了导数记号的那一侧：这种等式的右边才是答案，不算"整条方程"
_DERIV_LABEL = re.compile(
    r"^\s*(?:y|f\s*\(\s*x\s*\)|y\s*['′’]|dy\s*/\s*dx|d\s*y\s*/\s*d\s*x"
    r"|f\s*['′’]\s*\(\s*x\s*\))\s*$", re.I)


def _is_equation_answer(src) -> bool:
    """「答」是不是被写成了整条方程（左边有实义项，不是单纯的 y' = ...）。"""
    s = str(src or "")
    if "=" not in s:
        return False
    left = s.rsplit("=", 1)[0]
    if _DERIV_LABEL.match(left):
        return False
    return bool(re.search(r"[A-Za-z]", left))


def _build_nodes(raw_nodes: list, meta: dict, budget: int) -> list[dict]:
    """把模型给的原始节点洗成引擎可用的节点（清洗 + 兜底）。"""
    nodes = []
    for i, nd in enumerate(raw_nodes[:budget], start=1):
        # 兼容三种写法：新的「问」、旧的「提问」、更旧的「教学意图」
        ask = nd.get("问") or nd.get("提问") or nd.get("question") or nd.get("教学意图") or ""
        ans = nd.get("答") or nd.get("期望答案") or ""
        # 「式」是模型写的自由文本，必须先洗成可求值的表达式再入库：
        # 洗不出来就当没有数值答案（判定退回关键词匹配），绝不让一个
        # "二阶" 这种串被 sympify 悄悄当成符号，把答对的学生判成错
        raw_num = nd.get("式") or nd.get("sympy检查")
        numeric = mathcheck.clean_numeric(raw_num)
        if not numeric:
            numeric = mathcheck.clean_numeric(ans)     # 模型常把值写在「答」里
        # 「答」写成整条方程时（"y²+2xyy'-1/(x+1)+y'/y=0"），clean_numeric 只留
        # 最后一个等号右边，整条式子悄悄变成数值 0 —— 之后拿 0 去比对，答对的
        # 学生被判错。这种答案本来就该当表达式判，不能留下数值
        if numeric and (_is_equation_answer(raw_num) or _is_equation_answer(ans)):
            numeric = ""
        # 「式」与「答」互相矛盾时丢弃「式」：实测模型给过 答"y=9x-16" 配
        # 式"1/2" 的垃圾——不丢的话数值比对会劫持判定，学生答对也判错。
        # 宁可少一层判定，不可误判
        if numeric:
            ans_expr = mathcheck.parse_student_expr(ans) if ans else None
            num_val = mathcheck.parse_expr(numeric)
            if (ans_expr is not None and getattr(ans_expr, "free_symbols", set())
                    and num_val is not None and num_val.is_number
                    and not mathcheck.same(ans_expr, num_val)):
                numeric = ""
        # "1 和 3" 这类多值答案：把倒序说法也加进可接受答案（学生打 "3 和 1"）
        answers = [ans] if ans else []
        m2 = re.fullmatch(r"(-?\d+)\s*和\s*(-?\d+)", (ans or "").strip())
        if m2 and m2.group(1) != m2.group(2):
            answers.append(f"{m2.group(2)} 和 {m2.group(1)}")
        nodes.append({
            "n": i,
            "stage": _STAGES[min(i - 1, len(_STAGES) - 1)],
            "knowledge": nd.get("知识点") or (meta.get("知识点") or ["未分类"])[0],
            "question": ensure_question(_to_question(clean_ask(ask))),
            "expect": ans,
            "numeric": numeric,
            "answers": answers,
            "partial_kw": [],
            "correct_reply": "",
            "partial_hint": "",
            "hints": [],
        })
    return nodes


_IMPLICIT_RULES = """
这道题是隐函数求导，按标准解法固定这几步，一步一个节点：
1) 方程两边对 x 求导（含 y 的项求导后要带一个 y'）→ 得到求导后的等式
2) 把含 y' 的项归到一边，解出 y'（用 x 和 y 表示）
3) 把题给的点代回**原方程**，求出该点的 y
4) 把点代进 y' 的表达式，求出 dy/dx —— 这一节点答完题目就结束
禁止合并对数、通分、把方程改写成别的形式再求导；禁止跳过第 3 步直接代值。
**「问」里绝对不许写出该点的 y 值或最终答案**——那是要学生自己算出来的，
写进问句就等于替他做了。"""


def _implicit_hint(truth: dict) -> str:
    """把符号引擎算好的真相塞给模型当内部参考。

    小模型不是"不会拆"，是"算不对"——实测同一个提示词连跑三次，xy² 的导数
    分别写成 y²+2xyy'、2xy+y²y'、和一次对数合并后的怪式子。给它算好的值，
    它负责组织话术，数学对错由这里兜底。
    """
    return (
        "\n\n【服务端已用符号引擎算好（内部参考，用来校正你的拆解；"
        "禁止在「问」里说出这些值）】\n"
        f"- 该点的 y 值：y = {mathcheck.pretty(truth['y0'])}\n"
        f"- 最终结果：dy/dx = {mathcheck.pretty(truth['value'])}\n"
        "「问」里只出现题目给的条件（点 x=..., 原方程），这两条只用来核对你写的「答」。"
    )


def _issue_feedback(issues: list[str]) -> str:
    return ("\n\n你上一次输出的拆解有这些硬伤，逐条改掉后**重新输出全部节点**：\n- "
            + "\n- ".join(issues))


def _value_of(node: dict):
    """节点答案的数值形式；容忍 "ln y = 1 ⇒ y=e" 这种带推导的写法。"""
    num = node.get("numeric")
    if num:
        e = mathcheck.parse_expr(num)
        if e is not None:
            return e
    t = str(node.get("expect") or "").strip()
    if not t:
        return None
    for sep in ("⇒", "→", "=>", "所以", "即", "得"):
        if sep in t:
            t = t.rsplit(sep, 1)[-1]
    t = t.strip().strip("。；;")
    if "=" in t:
        t = t.rsplit("=", 1)[-1]
    return mathcheck.math_text_to_expr(t)


_DERIV_MARK = re.compile(r"y\s*['′’]|dy\s*/\s*d\s*x", re.I)


def _clause_kinds(text: str, truth: dict, talks_deriv: bool) -> tuple[set, bool]:
    """单句答案命中了哪几类（eq/dy/term），以及是否确实算错。

    容忍"xy² 的导数是 y²+2xyy'"、"所以 y' = …"这种带引导语的写法：
    逐个把"是/为/得/等于"后面的尾巴也拿来试一遍。
    """
    cands = [text]
    for sep in ("是", "为", "得", "等于", "：", ":"):
        if sep in text:
            cands.append(text.rsplit(sep, 1)[-1])
    talks_deriv = talks_deriv or "导" in text
    kinds: set[str] = set()
    parsed = False
    for c in cands:
        c = c.strip().strip("。；;")
        if not c:
            continue
        mentions = bool(_DERIV_MARK.search(c))
        eq = mathcheck.equation_diff_ok(c, truth)
        dy = mathcheck.dy_expr_ok(c, truth)
        if mentions:
            term = mathcheck.term_deriv_ok(c, truth)
        elif talks_deriv:
            term = mathcheck.x_term_deriv_ok(c, truth)
        else:
            term = None
        hit = False
        for name, r in (("eq", eq), ("dy", dy), ("term", term)):
            if r is True:
                kinds.add(name)
                hit = True
            elif r is not None:
                parsed = True       # 解析出来了但对不上
        if hit:
            return kinds, False
    # 一句话都没解析出来（整段是推导过程）时算 na，不算错——按错处理会误杀
    # 完全正确的计划，实测就发生过：带演算过程的正确答案被整份打回
    return kinds, parsed


def _node_kinds(expect: str, truth: dict, question: str = "") -> tuple[set, bool]:
    exp = str(expect or "").strip()
    if not exp:
        return set(), False
    parts = [p for p in re.split(r"[，,；;。]", exp) if p.strip()] or [exp]
    talks_deriv = "导" in str(question or "")
    kinds: set[str] = set()
    bad = False
    for p in parts:
        k, b = _clause_kinds(p, truth, talks_deriv)
        kinds |= k
        bad = bad or b
    return kinds, bad


def _node_math(expect: str, truth: dict, question: str = "") -> str:
    """这个节点的答案在数学上站不站得住：ok / bad / na。

    含 y' 的答案只认三类：两边求导后的等式、解出的 y'、F 中某一项的全导数。
    一句话里写了几段（逐项求导那种）就逐段判——命中任意一类就算过关，
    有一句解析出来却对不上才算算错。
    """
    kinds, bad = _node_kinds(expect, truth, question)
    if bad:
        return "bad"
    return "ok" if kinds else "na"


# 句子里"得 y=e"这种：已经把结果算出来给学生看了
_ASSIGN_CALC = re.compile(r"(?:得|即|所以|于是|因此|可知|⇒|→|=>)\s*[^，。；？]{0,24}[=＝]")
# 纯代值写法（x=0、y=2）不算泄题，先把它们剔掉再看还剩不剩等号
_SUBST = re.compile(r"[A-Za-z]\s*[=＝]\s*[-+]?[0-9A-Za-z]+")


# 「答」里出现这些，说明模型写的是推导/自我怀疑，不是答案
_BAD_EXPECT = re.compile(r"[？?]|≠|重新检查|重新算|不确定|似乎|实际上|自相矛盾")


def _scrub_leak(q: str, truth: dict, is_last: bool) -> str:
    """把问句里已经写出来的 y 值抹掉。

    小模型很爱写"把 x=0、y=e 代回原方程，求 y"——答案就这么漏了。整份计划
    打回重来代价太大（本地一次拆题十几秒），而且下一轮它还会这么写。这里
    只摘掉那一小段，句子仍然读得通，学生该算的还是得算。
    """
    if is_last:
        return q                      # 终值节点本来就要用到 y 的值
    y0 = truth.get("y0")
    if y0 is None:
        return q
    forms = [f for f in (truth.get("y0_s") or "", mathcheck.pretty(y0)) if f]
    out = q
    for f in forms:
        out = re.sub(rf"[、,，]?\s*y\s*[=＝]\s*{re.escape(f)}(?![A-Za-z0-9])", "", out,
                     flags=re.I)
    return re.sub(r"[、,，]\s*[，,]", "，", out)


def _leak_issues(nodes: list[dict], truth: dict) -> list[str]:
    """问句里不许把演算过程或答案写出来——写出来学生就没得算了。"""
    issues: list[str] = []
    eq = re.sub(r"\s+", "", truth.get("eq") or "")
    y0 = truth.get("y0")
    # y 的值可能被模型写成 e 或 SymPy 的 E，两种都算泄漏
    forms = [f for f in (str(truth.get("y0_s") or ""),
                         mathcheck.pretty(y0) if y0 is not None else "") if f]
    for n in nodes:
        q = n.get("question") or ""
        rest = re.sub(r"\s+", "", q)
        if eq:
            rest = rest.replace(eq, "")
        if _ASSIGN_CALC.search(rest) or re.search(r"[=＝]", _SUBST.sub("", rest)):
            issues.append(f"第 {n['n']} 句里写了等式/演算过程，等于把答案先说了")
        # y 的值一旦在问句里出现，这一段就白问了；终值节点除外（那时 y 已经求过）
        if n["n"] < nodes[-1]["n"] and forms:
            for f in forms:
                if len(f) == 1:
                    hit = re.search(rf"(?<![A-Za-z])[{f.lower()}{f.upper()}](?![A-Za-z])", rest)
                else:
                    hit = f in rest
                if hit:
                    issues.append(f"第 {n['n']} 句里已经写出了 y 的值 {f}，学生没得算")
                    break
    return issues


def _implicit_issues(nodes: list[dict], truth: dict) -> list[str]:
    """隐函数题的每一步都得算对，最后一步还得是最初要求的那个量。"""
    if not nodes:
        return ["一个节点都没有"]
    issues: list[str] = []
    deriv_hit = dy_hit = 0
    deriv_at = None
    y0_nodes: list[int] = []
    for n in nodes:
        exp = n.get("expect") or ""
        kinds, bad = _node_kinds(exp, truth, n.get("question") or "")
        if bad:
            issues.append(f"第 {n['n']} 步的求导结果算错了，和符号引擎对不上")
        elif kinds:
            if "dy" in kinds:
                dy_hit += 1
            if kinds & {"eq", "term"}:
                deriv_hit += 1
                deriv_at = deriv_at or n["n"]
        elif mathcheck.same_equation_ok(exp, truth) and not _DERIV_MARK.search(exp):
            issues.append(f"第 {n['n']} 步的答案还是原方程本身，没有做任何计算")
        v = _value_of(n)
        if v is not None and mathcheck.same(v, truth["y0"]):
            y0_nodes.append(n["n"])

    if not deriv_hit:
        issues.append("缺少「对两边求导」这一步（完整等式或逐项导数都行）")
    if not dy_hit:
        issues.append("缺少「解出 y'」这一步")
    if not y0_nodes:
        issues.append(f"缺少「把 x={truth['x0_s']} 代回原方程求 y」这一步，或求出的 y 值不对")
    if deriv_at and y0_nodes and min(y0_nodes) < deriv_at:
        issues.append("求 y 的那一步排在了「两边求导」前面，顺序不对")

    last = nodes[-1]
    q = (last.get("question") or "").replace(" ", "").lower()
    if not ("dy/dx" in q or "y'" in q or "y′" in q):
        issues.append("最后一个节点没有问出题目要求的 dy/dx")
    else:
        v = _value_of(last)
        if v is None:
            issues.append("最后一个节点没给出能比对的答案")
        elif not mathcheck.same(v, truth["value"]):
            issues.append(f"最后一个节点的答案算错了（正确结果是 {truth['value_s']}）")
    return issues + _leak_issues(nodes, truth)


def plan_issues(nodes: list[dict], truth: dict | None = None) -> list[str]:
    """这份计划有什么硬伤。空列表 = 可以发给学生。"""
    issues: list[str] = []
    for n in nodes:
        q = n["question"]
        marks = q.count("？") + q.count("?")
        if marks > 1:
            issues.append(f"第 {n['n']} 句里塞了 {marks} 个问题，一步只问一件事")
        exp = n.get("expect") or ""
        if not exp.strip():
            issues.append(f"第 {n['n']} 个节点没写「答」，判定时无从对照")
        elif _BAD_EXPECT.search(exp):
            # 实测模型把「答」写成"e^2-0+1=2≠1，此处需重新检查：其实……"这种
            # 自相矛盾的独白。这种答案没法判、也不能给学生看，必须打回重写。
            # 注意别按长度卡：带完整推导的长答案本身是可以用的
            issues.append(f"第 {n['n']} 个节点的「答」不是干净的结果"
                          "（混进了推导过程或自我怀疑的话），只写最终那个数或式子")
    if truth:
        issues += _implicit_issues(nodes, truth)
    return issues


def _implicit_meta(meta: dict) -> dict:
    return {
        "题型": meta.get("题型") or "隐函数求导",
        "知识点": meta.get("知识点") or ["链式法则", "隐函数求导"],
        "难度": meta.get("难度") or "中等",
    }


def _implicit_skeleton(truth: dict) -> list[dict]:
    """符号引擎担保的骨架：模型两次都没拆对时用它，宁可朴素，不可教错。"""
    x0, y0s, vs = truth["x0_s"], truth["y0_s"], truth["value_s"]
    # 问句是给学生看的：用学生写法（e 而不是 SymPy 的 E）；
    # 「答」保留 SymPy 形式，判定层两种写法都认
    y0_show = mathcheck.pretty(truth["y0"])
    raw = [
        ("把方程两边同时对 x 求导（y 是 y(x)，含 y 的项求导后要乘一个 y'）。求导后的等式是什么？",
         truth["G_s"] + " = 0", "", "链式法则"),
        ("把含 y' 的项归到一边、解出 y'（用 x 和 y 表示），y' 等于什么？",
         "y' = " + truth["dy_s"], "", "隐函数求导"),
        (f"把 x={x0} 代回原方程，这时 y 等于几？", y0s, y0s, "方程求解"),
        (f"把 x={x0}、y={y0_show} 代进 y' 的表达式，dy/dx 等于几？", vs, vs, "代入求值"),
    ]
    nodes = []
    for i, (q, a, num, kp) in enumerate(raw, start=1):
        nodes.append({
            "n": i,
            "stage": _STAGES[min(i - 1, len(_STAGES) - 1)],
            "knowledge": kp,
            "question": q,
            "expect": a,
            "numeric": mathcheck.clean_numeric(num) if num else "",
            "answers": [a],
            "partial_kw": [],
            "correct_reply": "",
            "partial_hint": "",
            "hints": [],
        })
    return nodes


def cloud_plan(problem: str, cancel=None) -> dict:
    budget = _plan_budget()
    # 隐函数求导这类题先由符号引擎算出真相：既当拆题的隐藏参考，也当校验的尺子。
    # 算不出来（不是这类题）就返回 None，后面一切照旧
    truth = _truth_of(problem)
    system = PLAN_SYSTEM + (_IMPLICIT_RULES if truth else "")
    user = f"题目：\n{problem}" + (_implicit_hint(truth) if truth else "")

    def bail(reason: str, err: Exception | None = None):
        """模型彻底指望不上时：有符号真值就走骨架，没有就把错误原样抛出去。"""
        if truth:
            log.warning("隐函数拆题改用符号骨架（%s）", reason[:120])
            return _pack(_implicit_skeleton(truth), _implicit_meta({}), truth)
        raise err if err is not None else ValueError(f"计划生成失败：{reason}")

    meta = {"题型": "", "知识点": [], "难度": ""}
    last_nodes: list[dict] = []
    issues: list[str] = []
    # 小模型经常混进个别废节点。合格就直接用；有硬伤（含数学算错）才带上
    # 整改清单重试一次——本地重试一次要多等几分钟，能不重试就不重试。
    for attempt in range(2):
        try:
            # 不设 max_tokens：实测这类小模型先写上千 token 的 reasoning 再出正文，
            # 上限会把正文拦腰截断（JSON 断在半截 → 解析失败 → 白白重试）。
            # 脱缰输出靠超时和降级兜底，不靠砍预算。
            data = llm.chat_json(system, user, cancel=cancel)
        except llm.LLMError as e:
            raw_text = str(e)
            if "JSON" not in raw_text:
                return bail(raw_text, e)
            # 整段 JSON 救不回来时，退回"按字段名捞问句"
            salvaged = _salvage_last_reply()
            if not salvaged:
                return bail(raw_text, e)
            raw_nodes, meta = salvaged, {"题型": "", "知识点": [], "难度": ""}
        else:
            raw_nodes = (data.get("节点") or data.get("节点计划") or data.get("nodes")
                         or data.get("node") or [])   # 实测小模型会写成 "node"
            meta = {"题型": data.get("题型", ""), "知识点": data.get("知识点", []),
                    "难度": data.get("难度", "")}
        last_nodes = _build_nodes(raw_nodes, meta, budget)
        good = [n for n in last_nodes if _real_question(n["question"])]
        if not good:
            issues = ["一个能直接作答的节点都没有：全是流程反问、非问句或教师视角的备注"]
        else:
            issues = ([] if len(good) * 2 >= len(last_nodes)
                      else ["节点里混着不能作答的废节点（流程反问 / 不是问句 / 教师备注）"])
            nodes = _renumber(good)
            if truth:
                last_n = nodes[-1]["n"]
                for n in nodes:
                    n["question"] = _scrub_leak(n["question"], truth, n["n"] == last_n)
            issues += plan_issues(nodes, truth)
            if not issues:
                return _pack(nodes, meta, truth)
        if attempt == 0:
            system += _issue_feedback(issues)

    if truth:
        log.warning("隐函数拆题两次未过校验（%s），改用符号骨架", "；".join(issues)[:150])
        return _pack(_implicit_skeleton(truth), _implicit_meta(meta), truth)
    # 重试仍有废节点：能剔就剔，一个不剩也只能硬着头皮用（判定层有关键词兜底）
    good = [n for n in last_nodes if _real_question(n["question"])]
    nodes = _renumber(good) if good else last_nodes
    if not nodes:
        raise ValueError("计划生成失败：节点为空")
    return _pack_plan(nodes, meta)


def _renumber(nodes: list[dict]) -> list[dict]:
    """剔掉废节点后重排 n 和 stage，保证进度条和阶段标签不串位。"""
    for i, n in enumerate(nodes, start=1):
        n["n"] = i
        n["stage"] = _STAGES[min(i - 1, len(_STAGES) - 1)]
    return nodes


def _pack_plan(nodes: list[dict], meta: dict) -> dict:
    return {
        "题型": meta.get("题型", ""),
        "知识点": meta.get("知识点", []),
        "难度": meta.get("难度", ""),
        "nodes": nodes,
    }


def _pack(nodes: list[dict], meta: dict, truth: dict | None = None) -> dict:
    """打包成计划。有符号真值时顺手把标准答案留给服务端报告页
    （以前自定义题的报告里这一格只能显示"—"）。"""
    plan = _pack_plan(nodes, meta)
    if truth is not None:
        plan["_answer_latex"] = mathcheck.latex(truth["value"])
    return plan


def _salvage_last_reply() -> list[dict]:
    """JSON 解析失败后，从模型最后一次回复里捞问句。"""
    raw = llm.last_reply()
    return extract_questions(raw) if raw else []


VARIANT_TRANSLATE_SYSTEM = """你是数学表述翻译器。把学生对变式题的回答翻译成一个 SymPy 表达式。只输出 JSON，不要解释。

输入三段：
【变式题】题目
【SymPy 标准答案】仅供对照记号风格，不许泄露给学生、也不用你判对错
【学生的回答】翻译对象（可能来自语音输入，写法口语化）

翻译规则：
- 语音口语先翻成符号："五倍x的四次方"=5*x**4、"3x方"=3*x**2、
  "x的三次方"=x**3、"负"=负号、中文数字=阿拉伯数字、"X"=x
- 分段描述（"x大于零时是…，x小于零时是…"）合成一个表达式，奇偶/符号类可用 sign(x)，
  绝对值用 Abs(x)
- 只翻译学生自己说的内容，绝不补全、绝不纠正；翻不出来（不是数学表达式）expr 给空串
- 变量只用 x；乘号写 *，幂用 **；纯 ASCII

输出：{"expr":"5*x**4*sign(x)"}"""


def _translate_variant_answer(sess, text: str):
    """把学生的口语/分段式回答翻译成 SymPy 表达式（判定仍由符号引擎做）。"""
    v = sess.variant or {}
    user = (f"【变式题】{v.get('problem', '')}\n"
            f"【SymPy 标准答案】{v.get('answer_plain', '')}\n"
            f"【学生的回答】{text}")
    try:
        data = llm.chat_json(VARIANT_TRANSLATE_SYSTEM, user, retries=2)
    except llm.LLMError as e:
        log.warning("会话 %s 变式回答翻译失败：%s", sess.id, str(e)[:120])
        return None
    expr = str(data.get("expr", "") or "").strip()
    if not expr:
        return None
    return mathcheck.parse_student_expr(expr)


# ---------------------------------------------------------------- 引擎

class Engine:
    """无状态的静态方法集合；所有会话状态在 Session 对象里。"""

    @staticmethod
    def create(problem: str = "", problem_id: str | None = None,
               user: str = "", mode: str = "", cancel=None) -> tuple[Session, dict]:
        forced_mock = mode == "mock"
        use_demo = forced_mock or not llm.online()
        plan_fallback = False
        if use_demo:
            prob = get_problem(problem_id)
            plan = demo_plan(prob)
            problem = prob["latex"]
            pid = prob["id"]
        else:
            if not (problem or "").strip():
                problem = get_problem(problem_id)["latex"]
            try:
                plan = cloud_plan(problem, cancel=cancel)
            except Exception:
                # 模型拆题失败（超时/JSON 连续失败/网络断）：内置题直接用
                # 它的讲解脚本兜底，别让流程死在起点；自定义题没有脚本可用，
                # 只能把错误带出去（路由层会转成 400 说明原因）
                if problem_id and problem_id != "custom":
                    prob = get_problem(problem_id)
                    plan = demo_plan(prob)
                    plan_fallback = True
                else:
                    raise
            pid = problem_id or "custom"

        sess = Session(
            id=uuid.uuid4().hex[:10],
            problem=problem,
            mode="mock" if forced_mock else llm.mode(),
            plan=plan,
            problem_id=pid,
            user=(user or "")[:64],
        )
        first = sess.nodes[0]
        sess.log("assistant", first["question"], stage=first["stage"], node=1)
        log.info("新会话 %s · 模式 %s · 题 %s · %d 节点%s",
                 sess.id, sess.mode, pid, len(sess.nodes),
                 "（拆题失败，内置脚本兜底）" if plan_fallback else "")
        return sess, {
            "session_id": sess.id,
            "mode": sess.mode,
            "offline_demo": use_demo,
            "plan_fallback": plan_fallback,
            "problem_id": pid,
            "problem": sess.problem,
            "meta": {
                "题型": plan.get("题型"),
                "知识点": plan.get("知识点"),
                "难度": plan.get("难度"),
            },
            "total_nodes": len(sess.nodes),
            "message": first["question"],
            "stage": first["stage"],
            "choices": choices_of(first),
            "progress": [1, len(sess.nodes)],
        }

    @staticmethod
    def answer(sess: Session, text: str) -> dict:
        if sess.finished:
            # progress 必给：前端无条件读 d.progress[0]，少了就是一个
            # "Cannot read properties of undefined" 的红条
            return {"finished": True, "message": "这道题已经走完了。要看看这次的学习报告吗？",
                    "next_action": "self_assessment", "choices": [],
                    "progress": [len(sess.nodes), len(sess.nodes)]}

        node = sess.nodes[sess.node_idx]
        sess.log("student", text)

        # 防绕过：讨答案时固定话术共情，把问题拆到最小，不记错、不推进
        if hits(text, BYPASS_PHRASES):
            log.info("会话 %s 讨答案，已拦下", sess.id)
            reply = (
                "不给。给了你只是在读我的推导，不是在做题。\n\n"
                "我们退到最小的一步——" + hint_for(node, 0)
            )
            sess.log("assistant", reply, stage=node["stage"], node=node["n"])
            return {
                "message": ensure_question(reply),
                "stage": node["stage"],
                "note": "",
                "verdict": "bypass",
                "node_done": False,
                "next_action": "continue",
                "choices": choices_of(node),
                "progress": [sess.node_idx + 1, len(sess.nodes)],
            }

        # 降维进行中：当前回答是在答"前置知识点"那道小题，不是主问题
        prereq = node.get("prerequisite")
        if sess.prereq_active and prereq:
            if sess.mode == "mock" or not llm.online():
                verdict, _, _ = verdict_offline(text, prereq)
            else:
                try:
                    verdict, _, _ = verdict_cloud(text, prereq, sess.history)
                except llm.LLMError:
                    verdict, _, _ = verdict_offline(text, prereq)
            sess.prereq_active = False
            back = "\n\n回到刚才那一步：" + node["question"]
            reply = (prereq["reply"] if verdict == "correct" else prereq["explain"]) + back
            sess.log("assistant", reply, stage=node["stage"], node=node["n"])
            return {
                "message": ensure_question(reply),
                "stage": node["stage"],
                "note": "",
                "verdict": verdict,
                "downgraded": True,
                "node_done": False,
                "next_action": "continue",
                "choices": [],
                "progress": [sess.node_idx + 1, len(sess.nodes)],
            }

        # 节点下面可以挂一个"追问"：主问题答对后不立刻推进，先追问一句，
        # 让学生自己把规律说出来（发现式），而不是由 AI 直接总结。
        followup = node.get("followup")
        in_followup = bool(followup and sess.followup_done)
        target = followup if in_followup else node

        if sess.mode == "mock" or not llm.online():
            verdict, got, reply = verdict_offline(text, target)
        else:
            try:
                verdict, reply, _model_stage = verdict_cloud(text, target, sess.history)
                got = mathcheck.number_in(text) if target.get("numeric") else None
            except llm.LLMError:
                # 模型抽风 / 网络/ JSON 连续解析失败：判定层降级成离线判定。
                # 绝不能把 LLMError 漏成 500——学生只会看到一句没头没脑的
                # "请求失败：Internal Server Error"
                verdict, got, reply = verdict_offline(text, target)
        # 阶段标签由代码按当前节点指定——模型的 stage 是自由发挥，
        # 实测会导致界面上回顾/启发/纠偏乱跳
        stage = node["stage"]
        log.info("会话 %s 节点 %d 判定 %s", sess.id, node["n"], verdict)

        # 表达式答案没有数值可抽，只能听判定模型的——2B 级模型会把完全正确的
        # 式子判成错（实测把 e-e³ 判 wrong，学生于是被反复重问同一句）。
        # 符号引擎能对上的直接翻成 correct：对错由算力说话，不由小模型说话
        if verdict != "correct" and symbolic_accepts(target, text, _truth_of(sess.problem)):
            log.info("会话 %s 节点 %d 判定被符号层翻正", sess.id, node["n"])
            verdict, reply = "correct", "对，就是这样。"

        # 判定说错、话术却在肯定（实测：符号答案 e-e³ 被数值闸抽残翻成
        # wrong，reply 却是"对，就是这样"）——学生被夸完再被重问同一句，
        # 永远走不到终点。以判定为准：肯定话术丢弃，走标准纠错路径。
        if verdict != "correct" and reply and _AFFIRM_REPLY.search(reply):
            reply = ""

        # 每轮的 SymPy 核对结果不再显示：对了是废话，错了等于报答案。
        # 符号引擎的验证力放两处更合适——报告页的变式题标准答案，与测试用例。
        note = ""

        before_wrong = sess.node_wrong
        next_action = "continue"
        advanced = False

        def advance(final_reply: str, kind: str = "correct") -> str:
            """通过当前节点，回复里接上下一个问题（或收尾）。"""
            nonlocal advanced, next_action, stage
            sess.results.append({
                "n": node["n"], "knowledge": node["knowledge"],
                "verdict": kind, "wrong": sess.node_wrong,
                "downgraded": sess.prereq_used,
            })
            sess.node_idx += 1
            sess.node_wrong = 0
            sess.followup_done = False
            sess.prereq_active = False
            sess.prereq_used = False
            advanced = True
            if sess.finished:
                next_action = "self_assessment"
                return (final_reply + "\n\n**这道题走完了。** 先别听我总结——"
                        "凭直觉说一句：这道题，你觉得自己掌握了几分？")
            nxt = sess.nodes[sess.node_idx]
            stage = nxt["stage"]
            return final_reply + "\n\n" + nxt["question"]

        def pending_choices() -> list:
            """下一句话问的是哪道题，就显示哪道题的选项。"""
            if sess.finished:
                return []
            if sess.prereq_active:
                return choices_of(node.get("prerequisite"))
            if sess.followup_done and not advanced:
                return choices_of(node.get("followup"))
            return choices_of(sess.nodes[sess.node_idx])

        if verdict == "correct":
            if followup and not sess.followup_done:
                # 第一步答对了，但先别夸完就走——把追问抛出去，节点不推进
                sess.followup_done = True
                reply = node["correct_reply"] + "\n\n" + followup["question"]
            else:
                reply = _affirm(reply, node, followup["reply"] if in_followup else "")
                reply = advance(reply)
        else:
            sess.node_wrong += 1
            if in_followup:
                if sess.node_wrong >= 2:
                    # 追问连错两次就直接给结论，别在这个小问题上卡死。
                    # followup["reply"] 本身就是以结论开头的话（"对，五阶。……"），
                    # 通用机制不能再套某一题的模板
                    reply = advance(followup["reply"], kind="revealed")
                else:
                    # 第一次答错：把追问原样再问一遍。追问是所有题共用的机制，
                    # 这里写死过"分母 x^5 展到几阶"——线代题的追问也会蹦出这句
                    reply = "再想想——" + followup["question"]
            elif (sess.node_wrong == MAX_WRONG_PER_NODE - 1
                  and prereq and not sess.prereq_used):
                # 第二次答错：先别急着给答案，退到这一步底下那块地基去
                sess.prereq_used = True
                sess.prereq_active = True
                reply = ("卡住的往往不是这一步，是它底下那块。**先退一步**：\n\n"
                         + strip_retreat_prefix(prereq["question"]))
            elif sess.node_wrong >= MAX_WRONG_PER_NODE:
                base = reply or hint_for(node, -1)
                reply = advance(
                    base + "\n\n这一步是我替你补上的，不是你推出来的——"
                           "**记住它，等会儿自评时你要想起这一处**。",
                    kind="revealed",
                )
            else:
                if sess.node_wrong > 1:
                    reply = hint_for(node, sess.node_wrong - 1)
                elif not reply:
                    if verdict == "partial" and node.get("partial_hint"):
                        reply = node["partial_hint"]
                    else:
                        reply = hint_for(node, 0)

        sess.log("assistant", reply, stage=stage, node=min(sess.node_idx + 1, len(sess.nodes)))

        if next_action == "self_assessment":
            _prefetch_variant(sess)   # 等学生挑分的那几秒里提前把变式题出了

        return {
            "message": ensure_question(reply),
            "stage": stage,
            "note": note,
            "verdict": verdict,
            "node_done": advanced,
            "next_action": next_action,
            "choices": pending_choices(),
            "progress": [min(sess.node_idx + 1, len(sess.nodes)), len(sess.nodes)],
        }

    # ------------------------------------------------------------ 自评

    @staticmethod
    def self_assess(sess: Session, score: int) -> dict:
        """自评收五分制（0~5），内部各档阈值也按分走。"""
        score = max(0, min(5, int(score)))
        sess.self_score = score

        wrong_nodes = [r for r in sess.results
                       if r["verdict"] in ("wrong", "revealed") or r["wrong"] > 0]
        if score >= 4 and wrong_nodes:
            names = []
            for r in wrong_nodes:
                if r["knowledge"] not in names:
                    names.append(r["knowledge"])
            kp_text = "」「".join(names)
            sess.warning = {
                "level": "red",
                "title": "虚假自信预警",
                "score": score,
                "reason": f"「{kp_text}」这一步没能一次说对",
                "message": (
                    f"你自评 **{score} / 5 分**，但「{kp_text}」这一步你没能一次说对。\n\n"
                    "看懂了 ≠ 会做了。等下给你一道同型题，**不给提示**，自己做完——那才算数。"
                ),
            }
        elif score <= 2 and not wrong_nodes:
            sess.warning = {
                "level": "green",
                "title": "你比自己以为的更接近",
                "score": score,
                "reason": "全程没有卡住",
                "message": f"你只给自己 **{score} 分**，可这道题你几乎一路答对。"
                           "真实水平大概在 4 分以上。",
            }
        else:
            sess.warning = {
                "level": "neutral",
                "title": "自评与表现一致",
                "score": score,
                "reason": "",
                "message": f"自评 {score} 分，与作答记录基本吻合。",
            }

        sess.log("system", sess.warning["message"], kind="warning")
        log.info("会话 %s 自评 %d → %s", sess.id, score, sess.warning["level"])
        return sess.warning

    # ------------------------------------------------------------ 变式题

    @staticmethod
    def make_variant(sess: Session) -> dict:
        prob = get_problem(sess.problem_id)
        problem, expr = _variant_fallback(sess, prob)
        ask = prob["variant"].get("ask") or "直接给出结果"
        prefetched = _pop_variant_prefetch(sess.id)
        if prefetched is not None:
            problem, expr = prefetched   # 学生挑分的那几秒里已经出好了
        elif llm.online() and sess.mode != "mock":
            try:
                problem, expr = cloud_variant(sess)
            except llm.LLMError:
                # 出题调用挂了就用内置变式题，流程不能断
                pass

        answer = mathcheck.evaluate(expr)
        # 云端变式题常给"极限号里的表达式"而不是极限值（出题 prompt 就是这么
        # 要求的）——题目是极限题时，程序替它取 x→0 的极限，并把表达式改写成
        # limit(...) 存回去，保证 judge_variant 判分时用的是同一个值
        if (answer is not None and getattr(answer, "free_symbols", set())
                and "lim" in (sess.problem or "").lower()):
            lv = mathcheck.limit_of(expr, "x", 0)
            if lv is not None:
                expr = f"limit({expr}, x, 0)"
                answer = lv
        if answer is None and llm.online() and sess.mode != "mock":
            # 双保险：模型给的题/表达式求不出值，整体退回兜底变式题。
            # 标准答案显示成 "?" 等于告诉学生这产品连自己都算不明白
            problem, expr = _variant_fallback(sess, prob)
            answer = mathcheck.evaluate(expr)
        sess.variant = {
            "problem": problem,
            "expr": expr,
            "ask": ask,
            "answer_latex": mathcheck.latex(answer) if answer is not None else "?",
            "answer_plain": mathcheck.pretty(answer) if answer is not None else "?",
        }
        log.info("会话 %s 变式题 · expr=%s · verified=%s",
                 sess.id, expr[:60], answer is not None)
        return {"problem": problem, "ask": ask, "verified": answer is not None}

    @staticmethod
    def judge_variant(sess: Session, text: str) -> dict:
        if not sess.variant:
            return {"error": "还没有出变式题"}
        want = mathcheck.evaluate(sess.variant["expr"])
        # 先整式解析，再退回抽数字：标准答案可能是 e²−e⁶ 这种常数式，
        # 只抽数字会把 "exp(2)-exp(6)" 抽成 2，学生答对也判错
        got = mathcheck.parse_student_expr(text)
        if got is None:
            got = mathcheck.number_in(text)
        # SymPy 解析不出来（语音输入的分段描述最典型："x大于零时是五倍x四次方，
        # x小于零时是负5倍x的四次方"）且在线：让模型**只做翻译**，把学生的回答
        # 翻成 SymPy 表达式，对错仍然由符号引擎判——模型不经手判定结果。
        # 含中文的回答优先走翻译：数字抽取对语音文本不可靠，容易抽错数字直接判错。
        if (want is not None and mathcheck.has_cjk(text)
                and llm.online() and sess.mode != "mock"):
            translated = _translate_variant_answer(sess, text)
            if translated is not None:
                got = translated
        correct = mathcheck.same(got, want)
        sess.variant_result = {
            "text": text,
            "correct": correct,
            "student": mathcheck.latex(got) if got is not None else "",
            "answer": mathcheck.latex(want) if want is not None else "?",
        }
        sess.log("student", text, kind="variant")

        if correct and sess.warning and sess.warning.get("level") == "red":
            sess.warning["cleared"] = True
            sess.warning["message"] += "\n\n变式题独立做对了——**红灯解除**，掌握度上调。"
        elif not correct:
            sess.warning = sess.warning or {}
            sess.warning["aggravated"] = True

        # 自评 vs 变式：把"你给自己打的分"和"你独立做出来的结果"摆在一起
        score = sess.self_score if sess.self_score is not None else 2
        actual = 5 if correct else 0
        gap = score - actual
        if correct and gap >= -1:
            conclusion = f"自评 {score} 分，变式独立做对——**你的自评是准的**，这次掌握是真的。"
        elif correct:
            conclusion = f"自评只有 {score} 分，却独立做对了——**你比自己以为的更强**，下次可以自信一点。"
        elif gap > 1:
            conclusion = f"自评 {score} 分，变式却做错了——**这次自评高估了 {gap} 分**，"\
                         "坐实了「被讲懂 ≠ 会做」。"
        else:
            conclusion = f"自评 {score} 分，变式也做错了——分数很诚实，"\
                         "**你知道自己还没掌握**，这比虚高好。"

        return {
            "correct": correct,
            "student": sess.variant_result["student"],
            "answer": sess.variant_result["answer"],
            "cleared": bool(sess.warning and sess.warning.get("cleared")),
            "self_score": score,
            "gap": gap,
            "conclusion": conclusion,
        }

    # ------------------------------------------------------------ 报告

    @staticmethod
    def radar(sess: Session) -> dict:
        """雷达图聚合数据：按知识点维度的掌握度。

        只统计"真的答过"的节点——没走到的节点记 0 会把整张图压成一朵花，
        看不出真实水平。走到的节点不足 3 个时前端退回条形图（雷达至少要三角形）。
        """
        rep = Engine.report(sess)
        answered = [b for b in rep["bars"] if b["note"] != "未涉及"]
        return {
            "session_id": sess.id,
            "problem": sess.problem,
            "labels": [b["kp"] for b in answered],
            "values": [b["score"] for b in answered],
            "notes": [b["note"] for b in answered],
            "weak": [b["kp"] for b in answered if b["weak"]],
            "total": rep["总体掌握度"],
            "answered": len(answered),
            "total_nodes": len(sess.nodes),
            "enough": len(answered) >= 3,
        }

    @staticmethod
    def report(sess: Session) -> dict:
        kp_order, kp_score, kp_note = [], {}, {}

        for node in sess.nodes:
            kp = node["knowledge"]
            if kp not in kp_score:
                kp_order.append(kp)
            res = next((r for r in sess.results if r["n"] == node["n"]), None)
            if res is None:
                continue
            if res["verdict"] == "correct" and res["wrong"] == 0:
                s, note = 100, "一次答对"
            elif res["verdict"] == "correct" and res.get("downgraded"):
                s, note = 55, f"降维到前置知识点后通过（答错 {res['wrong']} 次）"
            elif res["verdict"] == "correct":
                s, note = 60, f"提示后通过（答错 {res['wrong']} 次）"
            elif res["verdict"] == "revealed" and res.get("downgraded"):
                s, note = 35, "降维后仍未能自己推出，已给出结论"
            elif res["verdict"] == "revealed":
                s, note = 40, "未能自己推出，已给出结论"
            else:
                s, note = 50, "部分正确"
            kp_score[kp] = s
            kp_note[kp] = note

        if sess.variant_result:
            delta = 10 if sess.variant_result["correct"] else -15
            for kp in ("泰勒展开", "展开阶数"):
                if kp in kp_score:
                    kp_score[kp] = max(0, min(100, kp_score[kp] + delta))
                    if delta > 0 and "变式" not in kp_note[kp]:
                        kp_note[kp] += "；变式题独立答对 +10"

        bars = [
            {
                "kp": kp,
                "score": kp_score.get(kp, 0),
                "note": kp_note.get(kp, "未涉及"),
                "weak": kp_score.get(kp, 0) < 70,
            }
            for kp in kp_order
        ]

        total_wrong = sum(1 for r in sess.results if r["wrong"] > 0)
        return {
            "session_id": sess.id,
            "problem": sess.problem,
            "mode": sess.mode,
            "总体掌握度": round(sum(b["score"] for b in bars) / len(bars)) if bars else 0,
            "bars": bars,
            "薄弱点": [b["kp"] for b in bars if b["weak"]],
            "答错节点数": total_wrong,
            "虚假自信记录": (
                {
                    "自评": sess.self_score,
                    "依据": sess.warning["reason"],
                    "状态": "已解除（变式题独立答对）" if sess.warning.get("cleared") else "待验证",
                }
                if sess.warning and sess.warning.get("level") == "red" else None
            ),
            "自评一致性": sess.warning.get("title") if sess.warning else None,
            "变式题": (
                {
                    "题目": sess.variant["problem"],
                    "学生答案": sess.variant_result["student"],
                    "SymPy 标准答案": sess.variant_result["answer"],
                    "结果": "正确" if sess.variant_result["correct"] else "错误",
                }
                if sess.variant and sess.variant_result else None
            ),
            "标准答案（服务端留存，未下发）":
                sess.plan.get("_answer_latex") or "—",
        }


# ---------------------------------------------------------- 变式题预生成
# 学生在自评面板挑分的那几秒，正好够云端出变式题——最后一节点答完的瞬间
# 就后台开跑，自评提交时变式题已经在手，页面零等待。
class _VariantPrefetch:
    __slots__ = ("event", "value")

    def __init__(self):
        self.event = threading.Event()
        self.value = None


_PREFETCH: dict[str, _VariantPrefetch] = {}


def _prefetch_variant(sess: Session) -> None:
    if sess.id in _PREFETCH or not (llm.online() and sess.mode != "mock"):
        return
    pf = _VariantPrefetch()
    _PREFETCH[sess.id] = pf

    def _work():
        try:
            pf.value = cloud_variant(sess)   # 内含校验与同型关卡，失败给 None
        except Exception:
            pf.value = None
        pf.event.set()

    threading.Thread(target=_work, daemon=True).start()


def _pop_variant_prefetch(sid: str):
    pf = _PREFETCH.pop(sid, None)
    if pf is None:
        return None
    pf.event.wait(timeout=90)    # 还没出完就等它——和同步生成同价，不吃亏
    return pf.value


def _limit_marker(t: str) -> bool:
    """题面是不是极限题——变式题同型关卡用的判定标记。"""
    t = (t or "").lower()
    return "lim" in t or "极限" in t


def _variant_fallback(sess: Session, prob: dict) -> tuple[str, str]:
    """变式兜底：题库题用自带变式；**自定义题绝不跨题型**——

    get_problem() 对自定义题会兜到题库第一题（等价无穷小，极限题），直接拿来
    当求导题的变式等于当场露馅。改用计划最后一节点：让学生独立把收尾那一步
    再算一遍，标准答案就是该节点的数值答案——同型、可验证。
    """
    if prob.get("id") == sess.problem_id and "variant" in prob:
        v = prob["variant"]
        return v["latex"], v["表达式"]
    for n in reversed(sess.plan.get("nodes") or []):
        if n.get("numeric"):
            return (f"（独立重算）把原题的最后一步再独立算一遍：{n['question']}",
                    n["numeric"])
    v = prob["variant"]          # 计划里连一个数值节点都没有：沿用旧行为
    return v["latex"], v["表达式"]


def cloud_variant(sess: Session) -> tuple[str, str]:
    """让 LLM 出同型题，并给出可用于 SymPy 验证的**最终答案**表达式。

    关键语义：「sympy表达式」必须是这道题的**答案**——求导题给导函数、
    极限题给极限号里的式子（程序代求极限）、数值题给结果。以前 prompt 写死
    "程序会对它求 x→0 的极限"，导致求导题模型只交原函数（|x|^3 的变式交了
    abs(x)**4），学生答对的导函数被判错。
    注意导函数写法：|x|^n 的导数是 n*x**(n-1)*sign(x)——不能写成
    sign(Abs(x))（x<0 时那是 +1，符号就错了）。
    表达式求不出来就整体回退内置变式题——标准答案显示成 "?" 等于告诉学生
    这产品连自己都算不明白，宁可题面保守一点。
    """
    system = (
        "基于原题出一道同型变式题：换数字或换结构。题型、考点、求解目标、难度"
        "都必须与原题完全一致——**原题求什么，变式就求什么**（原题求 y'(0) "
        "就还求 y'(0)，原题求极限就还求极限）。禁止跨题型：原题是求导 / "
        "隐函数求导，绝不允许出成极限、积分题。"
        "只输出 JSON，不要代码块、不要解释：\n"
        '{"题目":"题面，公式用 $...$ 包住","sympy表达式":"这道题最终答案的 SymPy 表达式"}\n'
        "「题目」必须是一条完整的中文题面，含具体数字，照原题的样子写。\n"
        "「sympy表达式」硬要求（程序会直接执行它算标准答案，所以必须就是**答案本身**）：\n"
        "- 求导题：写导函数。例：x^3 的导数 → 3*x**2；|x|^4 的导数 → 4*x**3；"
        "|x|^5 的导数 → 5*x**4*sign(x)（用 sign(x)，别用 sign(Abs(x))）\n"
        "- 隐函数求导题（给具体点的导数值）：写那个数值。"
        "例：xy^2-ln(x+1)+ln y=2 求 y'(0) → exp(2)-exp(6)\n"
        "- 极限题：只写极限号里的式子，程序自己会求 x→0 的极限。"
        "例：(exp(2*x)-1-2*x)/x**2\n"
        "- 定积分/方程等其他题：写求值结果，可用 integrate(...)、solve(...)\n"
        "- 禁止 sympy. 前缀；幂用 **，乘号必须写 *，不要中文、不要等号、不要单位\n\n"
        "示例一（原题是 lim(x→0) (e^x-1-x)/x^2 时）：\n"
        '{"题目":"$\\\\lim_{x \\\\to 0}\\\\frac{e^{2x}-1-2x}{x^{2}}$",'
        '"sympy表达式":"(exp(2*x)-1-2*x)/x**2"}\n'
        "示例二（原题是求 f(x)=|x|^3 的导数时）：\n"
        '{"题目":"求函数 $f(x)=\\\\left|x\\\\right|^{5}$ 的导数",'
        '"sympy表达式":"5*x**4*sign(x)"}\n'
        "示例三（原题是 xy^2-ln(x+1)+ln y=1 求 y'(0) 时——还是隐函数求导，"
        "只是把右边的 1 换成 2）：\n"
        '{"题目":"已知 $xy^{2}-\\\\ln(x+1)+\\\\ln y=2$（$y=y(x)$），'
        '求 $\\\\left.\\\\frac{dy}{dx}\\\\right|_{x=0}$",'
        '"sympy表达式":"exp(2)-exp(6)"}'
    )
    user = (
        f"原题：{sess.problem}\n知识点：{sess.plan.get('知识点')}\n"
        "出一道换数字/换结构的同型题：题型、考点、求解目标照原题，别改题型。"
    )
    fallback_prob = get_problem(sess.problem_id)
    extra = ""
    for attempt in range(2):
        data = llm.chat_json(system, user + extra)
        problem = (data.get("题目") or "").strip()
        expr = _clean_variant_expr(data.get("sympy表达式") or "")
        # 隐函数求导的变式：答案由符号引擎按新题面现算，不采信模型的估计。
        # 实测模型把 xy^2-ln(x+1)+ln y=0 的答案写成 1-e^-1，真值是 0——
        # 标准答案错了，学生答对也会被判错
        truth = mathcheck.implicit_derivative(problem)
        if truth is not None:
            expr = truth["value_s"]
        # 小模型会把提示词里的占位符原样抄回来（实测输出过字面的"LaTeX题目"），
        # 题面不像人话就整体回退内置变式题
        if (_PLACEHOLDER.search(problem) or len(problem) < 12
                or not expr or mathcheck.evaluate(expr) is None):
            return _variant_fallback(sess, fallback_prob)
        # 同型关卡：原题不是极限题，变式却出成极限（或反过来）——
        # 实测隐函数求导题被出成极限变式。给一次带反馈的重修机会
        if _limit_marker(problem) != _limit_marker(sess.problem):
            if attempt == 0:
                extra = ("\n\n你上一道跑题型了：变式题的题型和求解目标必须与原题"
                         "完全一致，原题不是极限题就不要出极限题。重新输出。")
                continue
            return _variant_fallback(sess, fallback_prob)
        return problem, expr
    return _variant_fallback(sess, fallback_prob)


# 小模型会把提示词里的占位符原样抄回来当题面（实测输出过字面的"LaTeX题目"）
_PLACEHOLDER = re.compile(r"LaTeX|题面|占位|表达式\"|示例|例如")

def _clean_variant_expr(s: str) -> str:
    """洗模型给的 SymPy 表达式。

    常见毛病：写 `sympy.limit((...), x, 0)` 这种带前缀/函数调用的形式——
    沙箱命名空间里没有 `sympy` 这个名字，求值必然 NameError。剥掉前缀、
    从 limit(...) 调用里抠出内层表达式，再交给 evaluate 验证。
    """
    s = str(s or "").strip().strip("`$ ")
    s = s.replace("sympy.", "")
    m = re.search(
        r"^limit\s*\(\s*(.+?)\s*,\s*x\s*(?:,[^,()]*)?(?:,\s*dir\s*=\s*['\"]?[+-]['\"]?)?\s*\)\s*$",
        s,
    )
    if m:
        s = m.group(1)
    return s.strip()
