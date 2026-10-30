# -*- coding: utf-8 -*-
"""确定性计算层。

原则：任何数字出现在学生眼前之前，都必须经过这里算一遍。
LLM 只负责"怎么说"，不负责"等于几"。
"""
from __future__ import annotations

import re
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr as _parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

# 受限命名空间：只放数学函数，不放 __import__ / open / eval 可达的东西
_SAFE = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "cot": sp.cot,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt,
    "Abs": sp.Abs, "pi": sp.pi, "E": sp.E, "e": sp.E, "I": sp.I, "oo": sp.oo,
}

_CN_WORDS = [
    ("二分之一", sp.Rational(1, 2)),
    ("三分之一", sp.Rational(1, 3)),
    ("四分之一", sp.Rational(1, 4)),
    ("一半", sp.Rational(1, 2)),
    ("六分之一", sp.Rational(1, 6)),
    ("三分之一", sp.Rational(1, 3)),
]

_FRAC = re.compile(r"-?\d+\s*/\s*\d+")
_DEC = re.compile(r"-?\d+\.\d+")
_INT = re.compile(r"-?\d+")


def _context_ok(text: str, start: int, end: int) -> bool:
    """过滤掉 x^2、x2、2x、x² 这类"不是答案"的数字。"""
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    if (before and before in "^_") or (after and after in "^_"):
        return False
    if before.isascii() and before.isalpha():
        return False
    if after.isascii() and after.isalpha():
        return False
    return True


_FRAC_TEX = re.compile(r"\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}")
_FULL_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")

# 受限命名空间：只放数学对象，不放 __import__ / open / eval 可达的东西
_EVAL_NS = {
    "x": sp.Symbol("x"),
    "Symbol": sp.Symbol,
    "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "integrate": sp.integrate, "limit": sp.limit, "diff": sp.diff,
    "Matrix": sp.Matrix, "Rational": sp.Rational,
    "E": sp.E, "pi": sp.pi, "oo": sp.oo,
    "max": max, "min": min, "abs": abs,
}

# 求值前的黑名单：dunder 遍历、动态执行、反射访问
_DANGEROUS = re.compile(
    r"__|import\b|open\s*\(|eval\s*\(|exec\s*\(|compile\s*\(|"
    r"getattr|setattr|delattr|globals\s*\(|locals\s*\(|vars\s*\(|lambda\b"
)


def number_in(text: str):
    """从一段自然语言里抽出学生给出的最终数值。

    策略：分数/小数优先，整数只在没被分数吞掉时才算；取最靠右的一个
    （学生习惯把最终答案放在最后）。
    返回 sympy 表达式，抽不到返回 None。
    """
    text = (text or "").translate(_FULL_DIGITS)   # ２ → 2，否则 sympify 出来不是整数
    text = _FRAC_TEX.sub(r"\1/\2", text)          # \frac{1}{2} → 1/2

    for word, val in _CN_WORDS:
        if word in text:
            return val

    spans = []
    for pat in (_FRAC, _DEC):
        for m in pat.finditer(text):
            if _context_ok(text, m.start(), m.end()):
                spans.append((m.start(), m.end(), m.group().replace(" ", "")))
    occupied = [(s, e) for s, e, _ in spans]

    for m in _INT.finditer(text):
        if any(s <= m.start() < e for s, e in occupied):
            continue  # 落在分数/小数内部，跳过
        if _context_ok(text, m.start(), m.end()):
            spans.append((m.start(), m.end(), m.group()))

    if not spans:
        return None
    spans.sort(key=lambda t: t[0])
    raw = spans[-1][2]
    try:
        return sp.sympify(raw, rational=True)
    except Exception:
        return None


def same(a, b, tol: float = 1e-9) -> bool:
    """判断两个数学量是否严格相等（分数走符号化简，浮点走容差）。"""
    if a is None or b is None:
        return False
    try:
        diff = sp.simplify(sp.nsimplify(a) - sp.nsimplify(b))
        if diff == 0:
            return True
        if diff.is_number:
            return abs(complex(diff.evalf())) < tol
        return False
    except Exception:
        return False


def limit_of(expr_str: str, var: str = "x", point=0):
    """符号求极限。失败返回 None。"""
    try:
        x = sp.Symbol(var)
        expr = _parse_expr(expr_str, local_dict={**_SAFE, var: x}, transformations=TRANSFORMS)
        return sp.limit(expr, x, point)
    except Exception:
        return None


def parse_expr(s: str):
    """把 "1/2" 这种字符串解析成精确有理数。"""
    try:
        return sp.sympify(str(s), rational=True)
    except Exception:
        return None


# ---------------------------------------------------------------- 约束模型输出
#
# 节点的「期望值」是模型生成的，它写什么都可能：`1/2`、`\frac{1}{2}`、`二阶`、
# `y'=3x^2-3`、`约等于 0.5`。直接扔给 SymPy 的后果是两种：要么抛异常，
# 要么**悄悄解析成一个叫"二阶"的符号**——后者更糟，判对判错全凭运气。
# 所以在入库前先洗一遍：洗不出一个干净表达式就当"没有数值答案"，
# 判定退回关键词匹配，宁可少一层，也不要瞎判。

_MAX_EXPR = 80                       # 一条期望值不该比这更长
_ALLOWED_SYMBOLS = {"x", "t"}        # 题目里的自变量；别的符号一律视为噪声

_TEX_FRAC = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_TEX_SQRT = re.compile(r"\\sqrt\s*\{([^{}]*)\}")
_TEX_POW = re.compile(r"\^\s*\{([^{}]*)\}")
_TEX_SPACE = re.compile(r"\\left|\\right|\\!|\\,|\\;|\\:|\\quad|\\qquad")
_TEX_CMD = [
    (r"\cdot", "*"), (r"\times", "*"), (r"\div", "/"),
    (r"\pi", "pi"), (r"\infty", "oo"),
    (r"\arcsin", "asin"), (r"\arccos", "acos"), (r"\arctan", "atan"),
    (r"\ln", "log"), (r"\log", "log"), (r"\exp", "exp"),
    (r"\sin", "sin"), (r"\cos", "cos"), (r"\tan", "tan"), (r"\cot", "cot"),
]
# 汉字、中文标点、全角符号：出现就说明这句里有"话"，不是表达式
_CJK = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff01-\uff5e]")
_EXPR_CHARS = re.compile(r"^[0-9A-Za-z_+\-*/()., ]+$")
_FULL = str.maketrans("０１２３４５６７８９（）＋－＊／＝＾", "0123456789()+-*/=^")


def clean_numeric(raw) -> str:
    """把模型给的"期望值"洗成能被 SymPy 求值的表达式；洗不出来返回空串。

    接受：`0`、`1/2`、`-16`、`0.5`、`Rational(1,2)`、`sqrt(2)/2`、`3*x^2-3`、
          `$\\frac{1}{2}$`、`y'=3x^2-3`（取等号右边）
    拒绝：`二阶`、`x = 2 时`、`约等于 0.5`、`2cm`、`3x²-3`（上标字符）、超长串

    返回值是规范化后的 SymPy 文本（`0.5` → `1/2`），保证 `parse_expr` 一定能解析。
    """
    s = str(raw or "").strip()
    if not s:
        return ""
    s = s.translate(_FULL)
    s = s.strip("$`'\"").strip()
    # 只要一个值：`y'=3x^2-3` 取最后一段等号右边
    if "=" in s:
        s = s.rsplit("=", 1)[-1].strip()
    if not s:
        return ""

    # 先处理带花括号的 LaTeX，再处理其余命令
    for _ in range(4):                       # 允许有限层嵌套
        new = _TEX_FRAC.sub(r"((\1)/(\2))", s)
        new = _TEX_SQRT.sub(r"sqrt(\1)", new)
        new = _TEX_POW.sub(r"**(\1)", new)
        if new == s:
            break
        s = new
    s = _TEX_SPACE.sub("", s)
    for cmd, rep in _TEX_CMD:
        s = s.replace(cmd, rep)
    s = s.replace("^", "**").replace(" ", "")
    s = s.strip("。；;")

    if not s or len(s) > _MAX_EXPR:
        return ""
    if _CJK.search(s) or not _EXPR_CHARS.match(s):
        return ""                            # 有汉字、撇号、花括号残留……
    if _DANGEROUS.search(s):
        return ""

    expr = _as_expr(s)
    if expr is None:
        return ""
    if not isinstance(expr, sp.Basic):
        return ""
    # 关系式 / 布尔 / 集合不是"一个值"，别当答案用
    if (getattr(expr, "is_Boolean", False) or getattr(expr, "is_Relational", False)
            or isinstance(expr, sp.Set)):
        return ""
    free = {str(sym) for sym in getattr(expr, "free_symbols", set())}
    if free - _ALLOWED_SYMBOLS:
        return ""                            # `2cm` 会变成一个叫 cm 的符号
    # 未求值的积分/极限留给题目自己的 evaluate()，别在这里当答案
    if expr.has(sp.Integral, sp.Limit, sp.Derivative, sp.Sum, sp.Product):
        return ""
    return sp.sstr(expr)


def _as_expr(s: str):
    """把清洗过的字符串变成 sympy 表达式。

    两条路都试，优先要"符号干净"的那个：
    1. `sympify(rational=True)` —— 精确，`1/2` 是 Rational 而不是 0.5
    2. `parse_expr` + 隐式乘法 —— 兜住模型爱写的 `3x^2-3`（少个乘号），
       顺带把 `e` `pi` 这类落进 `_SAFE` 的常量认对
       （`sympify("e**2")` 会把 e 当成一个普通符号，第二条路才知道它是自然常数）
    """
    cands = []
    try:
        cands.append(sp.sympify(s, rational=True))
    except Exception:
        pass
    try:
        ns = {**_SAFE, "x": sp.Symbol("x"), "t": sp.Symbol("t")}
        cands.append(_parse_expr(s, local_dict=ns, transformations=TRANSFORMS))
    except Exception:
        pass
    cands = [c for c in cands if isinstance(c, sp.Basic)]
    if not cands:
        return None
    for c in cands:
        if {str(sym) for sym in getattr(c, "free_symbols", set())} <= _ALLOWED_SYMBOLS:
            return c
    return cands[0]


def evaluate(expr_str: str):
    """在受限命名空间里求值一个 SymPy 表达式，返回符号结果。

    三层防护：
      1. 字符串层面先挡掉 dunder 与动态执行关键字（`().__class__.__bases__[0]
         .__subclasses__()` 这种遍历在纯命名空间隔离下是能跑通的）
      2. 不给 `__builtins__`
      3. 命名空间里只有数学对象，没有任何 import 通道

    目前只用于我们自己写死的题库表达式；云端若要执行 LLM 生成的代码，
    还得再加超时与子进程隔离（大纲第三节第 3 条）。
    """
    src = str(expr_str)
    if _DANGEROUS.search(src):
        return None
    try:
        return eval(src, {"__builtins__": {}}, dict(_EVAL_NS))
    except Exception:
        return None


def latex(expr) -> str:
    """转成前端 KaTeX 能渲染的 LaTeX。"""
    try:
        return sp.latex(expr)
    except Exception:
        return str(expr)


def plain(expr) -> str:
    try:
        return sp.sstr(expr)
    except Exception:
        return str(expr)
