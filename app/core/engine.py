# -*- coding: utf-8 -*-
"""教学引擎：苏格拉底式追问 + 自评校准。

设计原则（对应大纲「Prompt 护栏」一节）：
  1. 流程由代码状态机控制 —— 推不推进节点、第几次答错给哪级提示、
     什么时候弹自评，全是 Python 的 if/else，不指望 LLM 自觉。
  2. 答案由 SymPy 决定 —— 学生报出的数值先过符号引擎，LLM 的判定可以被覆盖。
  3. 标准答案只在服务端 —— 下发给前端的永远只有"下一步问什么"。
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from app.core import llm, mathcheck
from app.core.judge import BYPASS_PHRASES, hint_for, hits, verdict_cloud, verdict_offline
from app.data.problems import PROBLEMS, get as get_problem

MAX_WRONG_PER_NODE = 3



def choices_of(node: dict | None) -> list:
    """节点可选的选择题选项；没有就是空列表，前端据此决定要不要画选项条。"""
    return list((node or {}).get("choices") or [])


def ensure_question(text: str) -> str:
    """最后一道保险：保证每条回复都以问句收尾。

    大纲把这条列为不可违反的硬规则，但提示文案是数据、云端话术是模型生成的，
    两边都可能漏。与其相信它们，不如在出口统一补一句——学生永远不会面对
    一条没法回应的陈述。
    """
    t = (text or "").rstrip()
    if not t:
        return "再想一想？"
    if t.endswith(("?", "？")):
        return t
    return t + "\n\n那你觉得呢？"


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


PLAN_SYSTEM = """你是资深数学助教，把一道数学题拆成教学节点。只输出 JSON，不要代码块，不要解释。

{"题型":"","知识点":[""],"难度":"","节点":[{"问":"","答":"","式":""}]}

字段说明：
- 问 = 直接对学生说的一句问句，以问号结尾。不要写"让学生…"这种教师备注。
- 答 = 学生该回答的简短要点（一个数或一个式子），给学生看的说法
- 式 = 这一小步期望值的**机器可读形式**，会被程序直接喂给 SymPy 求值。

「式」的硬要求（写错这一格，判定就失效）：
- 只能写一个表达式，不含汉字、不含等号左边、不含单位、不含说明文字、不含 $ 和 LaTeX
- 只用 ASCII：幂写 `**` 或 `^`，乘号写 `*`，分数写 `1/2`，根号写 `sqrt(2)`
- 正确：0 / 1/2 / -16 / 0.5 / 3*x^2-3 / sqrt(2)/2 / Rational(1,2) / exp(2)
- 错误：二阶 / x=2 时 / 约等于 0.5 / \\frac{1}{2} / 3x²-3 / 2 阶（展到第 2 阶得到 4）
- 这一小步的答案不是数值（比如问"用哪种方法"）时，写空串 ""，不要猜

要求：
1. 按真实解题顺序，每个节点只走一小步，不要合并。
2. 每问必须能用一个数、一个式子或一个明确名词回答，不要出开放语义题。
3. 只要「节点」，不要写完整解答——学生看不到，别浪费输出。
4. 节点数：__BUDGET__ 个以内。

示例（照这个格式写）：
{"题型":"极限计算","知识点":["泰勒展开"],"难度":"中等","节点":[{"问":"$e^{0}$ 等于几？","答":"1","式":"1"},{"问":"分子展开到第几阶？","答":"2","式":"2"}]}"""

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


def cloud_plan(problem: str) -> dict:
    budget = _plan_budget()
    user = f"题目：\n{problem}\n\n节点数不超过 {budget} 个。"
    try:
        data = llm.chat_json(PLAN_SYSTEM.replace("__BUDGET__", str(budget)), user)
        raw_nodes = data.get("节点") or data.get("节点计划") or data.get("nodes") or []
        meta = {"题型": data.get("题型", ""), "知识点": data.get("知识点", []),
                "难度": data.get("难度", "")}
    except llm.LLMError as e:
        # 整段 JSON 救不回来时，退回"按字段名捞问句"
        raw_text = str(e)
        raw_nodes, meta = [], {"题型": "", "知识点": [], "难度": ""}
        if "JSON" not in raw_text:
            raise
        salvaged = _salvage_last_reply()
        if not salvaged:
            raise
        raw_nodes, meta = salvaged, meta

    raw_nodes = raw_nodes[:budget]
    nodes = []
    for i, nd in enumerate(raw_nodes, start=1):
        # 兼容三种写法：新的「问」、旧的「提问」、更旧的「教学意图」
        ask = nd.get("问") or nd.get("提问") or nd.get("question") or nd.get("教学意图") or ""
        ans = nd.get("答") or nd.get("期望答案") or ""
        # 「式」是模型写的自由文本，必须先洗成可求值的表达式再入库：
        # 洗不出来就当没有数值答案（判定退回关键词匹配），绝不让一个
        # "二阶" 这种串被 sympify 悄悄当成符号，把答对的学生判成错
        numeric = mathcheck.clean_numeric(nd.get("式") or nd.get("sympy检查"))
        if not numeric:
            numeric = mathcheck.clean_numeric(ans)     # 模型常把值写在「答」里
        nodes.append({
            "n": i,
            "stage": _STAGES[min(i - 1, len(_STAGES) - 1)],
            "knowledge": nd.get("知识点") or (meta.get("知识点") or ["未分类"])[0],
            "question": ensure_question(clean_ask(ask)),
            "expect": ans,
            "numeric": numeric,
            "answers": [ans] if ans else [],
            "partial_kw": [],
            "correct_reply": "",
            "partial_hint": "",
            "hints": [],
        })
    if not nodes:
        raise ValueError("计划生成失败：节点为空")
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


# ---------------------------------------------------------------- 引擎

class Engine:
    """无状态的静态方法集合；所有会话状态在 Session 对象里。"""

    @staticmethod
    def create(problem: str = "", problem_id: str | None = None) -> tuple[Session, dict]:
        use_demo = not llm.online()
        if use_demo:
            prob = get_problem(problem_id)
            plan = demo_plan(prob)
            problem = prob["latex"]
            pid = prob["id"]
        else:
            if not (problem or "").strip():
                problem = get_problem(problem_id)["latex"]
            plan = cloud_plan(problem)
            pid = problem_id or "custom"

        sess = Session(
            id=uuid.uuid4().hex[:10],
            problem=problem,
            mode=llm.mode(),
            plan=plan,
            problem_id=pid,
        )
        first = sess.nodes[0]
        sess.log("assistant", first["question"], stage=first["stage"], node=1)
        return sess, {
            "session_id": sess.id,
            "mode": llm.mode(),
            "offline_demo": use_demo,
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
            if not llm.online():
                verdict, _, _ = verdict_offline(text, prereq)
            else:
                verdict, _, _ = verdict_cloud(text, prereq, sess.history)
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

        if not llm.online():
            verdict, got, reply = verdict_offline(text, target)
            stage = node["stage"]
        else:
            verdict, reply, stage = verdict_cloud(text, target, sess.history)
            got = mathcheck.number_in(text) if target.get("numeric") else None

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
                    # 追问连错两次就直接给结论，别在这个小问题上卡死
                    reply = advance(
                        f"分母是 $x^{{{followup['numeric']}}}$，就展到 ${followup['numeric']}$ 阶。"
                        + followup["reply"],
                        kind="revealed",
                    )
                else:
                    reply = (
                        "只看分母：$x^{5}$ 的最高次是几？"
                        "分子要含到 $x^{5}$ 这一项，展开到第几阶就够？"
                    )
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
        return sess.warning

    # ------------------------------------------------------------ 变式题

    @staticmethod
    def make_variant(sess: Session) -> dict:
        prob = get_problem(sess.problem_id)
        v = prob["variant"]
        problem, expr = v["latex"], v["表达式"]
        ask = v.get("ask") or "直接给出结果"
        if llm.online():
            problem, expr = cloud_variant(sess)

        answer = mathcheck.evaluate(expr)
        sess.variant = {
            "problem": problem,
            "expr": expr,
            "ask": ask,
            "answer_latex": mathcheck.latex(answer) if answer is not None else "?",
            "answer_plain": mathcheck.plain(answer) if answer is not None else "?",
        }
        return {"problem": problem, "ask": ask, "verified": answer is not None}

    @staticmethod
    def judge_variant(sess: Session, text: str) -> dict:
        if not sess.variant:
            return {"error": "还没有出变式题"}
        want = mathcheck.evaluate(sess.variant["expr"])
        got = mathcheck.number_in(text)
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
    """让 LLM 出同型题，并给出可用于 SymPy 验证的表达式。"""
    system = (
        "出一道具同型变式题，难度与原题相当，只输出 JSON："
        '{"题目":"LaTeX 题目","sympy表达式":"可直接用 sympy.limit 求值的 Python 表达式，变量 x"}'
    )
    user = (
        f"原题：{sess.problem}\n知识点：{sess.plan.get('知识点')}\n"
        "请出一道换数字/换结构的同型题，别改题型。"
    )
    data = llm.chat_json(system, user)
    fallback = get_problem(sess.problem_id)["variant"]
    return data.get("题目", fallback["latex"]), data.get("sympy表达式", fallback["表达式"])
