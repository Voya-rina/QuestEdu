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
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from app.core import llm, mathcheck
from app.core.judge import BYPASS_PHRASES, hint_for, hits, verdict_cloud, verdict_offline
from app.data.problems import PROBLEMS, get as get_problem

log = logging.getLogger(__name__)

MAX_WRONG_PER_NODE = 3



def choices_of(node: dict | None) -> list:
    """节点可选的选择题选项；没有就是空列表，前端据此决定要不要画选项条。"""
    return list((node or {}).get("choices") or [])


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

每个节点的规则：
1. 「问」：一小步的具体**疑问句**——句子里要有疑问词（多少 / 几 / 什么 / 哪一项），
   以？结尾，答案是一个数、一个简短结果或一个方法名。
   - 从"把具体值代进去看看"这类最基础的一步开始，先观察、再定方法、最后算结果
   - **一步只问一件事**：禁止一句话塞两个问题（"A 是什么，B 是什么？"）。
     需要两个信息就拆成两步；或者由题面固定其一，只问另一个
     （错误示例："选 u 是什么，选 dv 是什么？"；
      正确做法："令 u=x，那么 v=∫e^x dx 等于多少？"）
   - **不许把答案喂到嘴边**：问句里可以铺垫背景，但不能直接给出决定性的
     中间结果再问"剩下什么"（错误示例："展开到二阶是 1+x+x^2/2，那么分子
     约掉后剩下什么？"——展开式就是答案的一半）
   - **节点之间不许重复**：每个节点问不同的东西、有不同的答案；
     "特征值是几"和"两个特征值分别是多少"是同一个问题
   - 问句里的记号必须是学生**能打出来**的：别让学生输入 λ、∫ 这类符号——
     问"特征值"就问"λ 等于几"并让它答数字，行列式展开这类带 λ 的中间式
     不要当答案（「式」留空、靠下一问收数字）
   - 禁止指令式陈述："写出…""计算…""解出…"——改成"…是多少？"
   - 禁止让学生默写完整公式（如"写出 e^x 的泰勒展开式"）——改问其中的具体量：第几阶、系数是多少、剩哪一项
   - 禁止流程反问："你想…吗""准备好了吗""然后呢""那你觉得呢"
2. 「答」：期望要点，要能被学生**一口气说出来**——一个名词、一个数或一个式子；
   禁止 "A=…，B=…" 这种逗号并列的形式（没法作答，也没法判定）。
3. 「式」：「答」的 SymPy 形式：纯 ASCII（1/2、-16、sqrt(2)/2、exp(2)、3*x^2-3）；
   答案不是数值（方法名、名词）就写 ""。

节点数不超过 __BUDGET__ 个。

示例一（lim(x→0) (e^x-1-x)/x^2）：
{"题型":"极限计算","知识点":["泰勒展开"],"难度":"中等","节点":[
{"问":"先别急着算。把 x=0 代进分子和分母，它们各自变成几？","答":"分子分母都是 0","式":"0"},
{"问":"这种分子分母同时为 0 的极限形式叫什么？","答":"0/0 型未定式","式":""},
{"问":"分母是 x^2，分子用泰勒展开要展到第几阶？","答":"2 阶","式":"2"},
{"问":"代回分子化简后约掉 x^2，极限等于多少？","答":"1/2","式":"1/2"}]}

示例二（∫[0,1] x·e^x dx）：
{"题型":"定积分计算","知识点":["分部积分法"],"难度":"中等","节点":[
{"问":"被积函数是两种不同类型函数的乘积，这种积分用哪个方法？","答":"分部积分","式":""},
{"问":"令 u=x，那么 v=∫e^x dx 等于多少？","答":"e^x","式":"exp(x)"},
{"问":"代回分部积分公式，原函数是 x*e^x-e^x。代入上下限，结果是几？","答":"1","式":"1"}]}"""

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
    """节点数上限：本地模型慢，少拆几步；云端可以细一点。"""
    return 3 if llm.mode() == "local" else 6


# 流程性反问 / 开场白：学生没法回答，出现即说明节点是废的。
# 这是实测里小模型最常犯的毛病——第一个"问"就写成"那你觉得呢？"，
# 引擎的 ensure_question 只会往后面再补一个问号，越补越糊涂
_META_Q = re.compile(
    r"你想|你觉得|你认不认为|要不要|好了吗|准备好了|可以开始|我们先|先从(哪个|哪里)"
    r"|继续(吗|吧|么)|好不好|行不行|然后呢|怎么办|明白了吗|听懂了吗|接下来"
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
    return True


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
        numeric = mathcheck.clean_numeric(nd.get("式") or nd.get("sympy检查"))
        if not numeric:
            numeric = mathcheck.clean_numeric(ans)     # 模型常把值写在「答」里
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


def cloud_plan(problem: str, cancel=None) -> dict:
    budget = _plan_budget()
    system = PLAN_SYSTEM.replace("__BUDGET__", str(budget))
    user = f"题目：\n{problem}\n\n节点数不超过 {budget} 个。"

    data = raw_nodes = None
    meta = {"题型": "", "知识点": [], "难度": ""}
    last_nodes: list[dict] = []
    # 小模型经常混进个别废节点。过半合格就直接用、剔掉废节点（重试一次
    # 本地要多等几分钟，能不重试就不重试）；过半是废的才带着罪名重试一次。
    for attempt in range(2):
        try:
            # 不设 max_tokens：实测这类小模型先写上千 token 的 reasoning 再出正文，
            # 上限会把正文拦腰截断（JSON 断在半截 → 解析失败 → 白白重试）。
            # 脱缰输出靠超时和降级兜底，不靠砍预算。
            data = llm.chat_json(system, user, cancel=cancel)
        except llm.LLMError as e:
            # 整段 JSON 救不回来时，退回"按字段名捞问句"
            raw_text = str(e)
            if "JSON" not in raw_text:
                raise
            salvaged = _salvage_last_reply()
            if not salvaged:
                raise
            raw_nodes, meta = salvaged, {"题型": "", "知识点": [], "难度": ""}
        else:
            raw_nodes = (data.get("节点") or data.get("节点计划") or data.get("nodes")
                         or data.get("node") or [])   # 实测小模型会写成 "node"
            meta = {"题型": data.get("题型", ""), "知识点": data.get("知识点", []),
                    "难度": data.get("难度", "")}
        last_nodes = _build_nodes(raw_nodes, meta, budget)
        good = [n for n in last_nodes if _real_question(n["question"])]
        if good and len(good) * 2 >= len(last_nodes):
            return _pack_plan(_renumber(good), meta)
        if attempt == 0:
            system += ('\n\n你上一次输出的节点不合格：出现了流程性反问（如「你想…吗」「然后呢」）、'
                       '不是问句、或教师视角的备注。每个「问」必须是学生能直接作答的具体数学问题，'
                       '以问号结尾，包含作答所需的全部信息。重新输出全部节点。')

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
                if not reply:
                    reply = followup["reply"] if in_followup else node["correct_reply"]
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
        v = prob["variant"]
        problem, expr = v["latex"], v["表达式"]
        ask = v.get("ask") or "直接给出结果"
        if llm.online() and sess.mode != "mock":
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
            # 双保险：模型给的题/表达式求不出值，整体退回内置变式题。
            # 标准答案显示成 "?" 等于告诉学生这产品连自己都算不明白
            v0 = prob["variant"]
            problem, expr = v0["latex"], v0["表达式"]
            answer = mathcheck.evaluate(expr)
        sess.variant = {
            "problem": problem,
            "expr": expr,
            "ask": ask,
            "answer_latex": mathcheck.latex(answer) if answer is not None else "?",
            "answer_plain": mathcheck.plain(answer) if answer is not None else "?",
        }
        log.info("会话 %s 变式题 · expr=%s · verified=%s",
                 sess.id, expr[:60], answer is not None)
        return {"problem": problem, "ask": ask, "verified": answer is not None}

    @staticmethod
    def judge_variant(sess: Session, text: str) -> dict:
        if not sess.variant:
            return {"error": "还没有出变式题"}
        want = mathcheck.evaluate(sess.variant["expr"])
        if want is not None and getattr(want, "free_symbols", set()):
            # 标准答案是表达式（导函数这类）：把学生输入整体当表达式解析后
            # 符号比对（same() 会先化简、证不出再代点抽查）。
            # 以前只会抽数字——学生答 4x**3 会被抽出个 3 来比，必错
            got = mathcheck.parse_student_expr(text)
        else:
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
        "基于原题出一道同型变式题：换数字或换结构，题型、考点、难度都不变。"
        "只输出 JSON，不要代码块、不要解释：\n"
        '{"题目":"题面，公式用 $...$ 包住","sympy表达式":"这道题最终答案的 SymPy 表达式"}\n'
        "「题目」必须是一条完整的中文题面，含具体数字，照原题的样子写。\n"
        "「sympy表达式」硬要求（程序会直接执行它算标准答案，所以必须就是**答案本身**）：\n"
        "- 求导题：写导函数。例：x^3 的导数 → 3*x**2；|x|^4 的导数 → 4*x**3；"
        "|x|^5 的导数 → 5*x**4*sign(x)（用 sign(x)，别用 sign(Abs(x))）\n"
        "- 极限题：只写极限号里的式子，程序自己会求 x→0 的极限。"
        "例：(exp(2*x)-1-2*x)/x**2\n"
        "- 定积分/方程等其他题：写求值结果，可用 integrate(...)、solve(...)\n"
        "- 禁止 sympy. 前缀；幂用 **，乘号必须写 *，不要中文、不要等号、不要单位\n\n"
        "示例一（原题是 lim(x→0) (e^x-1-x)/x^2 时）：\n"
        '{"题目":"$\\\\lim_{x \\\\to 0}\\\\frac{e^{2x}-1-2x}{x^{2}}$",'
        '"sympy表达式":"(exp(2*x)-1-2*x)/x**2"}\n'
        "示例二（原题是求 f(x)=|x|^3 的导数时）：\n"
        '{"题目":"求函数 $f(x)=\\\\left|x\\\\right|^{5}$ 的导数",'
        '"sympy表达式":"5*x**4*sign(x)"}'
    )
    user = (
        f"原题：{sess.problem}\n知识点：{sess.plan.get('知识点')}\n"
        "出一道换数字/换结构的同型题，别改题型。"
    )
    data = llm.chat_json(system, user)
    fallback = get_problem(sess.problem_id)["variant"]
    problem = (data.get("题目") or "").strip()
    expr = _clean_variant_expr(data.get("sympy表达式") or "")
    # 小模型会把提示词里的占位符原样抄回来（实测输出过字面的"LaTeX题目"），
    # 题面不像人话就整体回退内置变式题
    if (_PLACEHOLDER.search(problem) or len(problem) < 12
            or not expr or mathcheck.evaluate(expr) is None):
        return fallback["latex"], fallback["表达式"]
    return problem, expr


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
