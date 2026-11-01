# -*- coding: utf-8 -*-
"""确定性计算层。

原则：任何数字出现在学生眼前之前，都必须经过这里算一遍。
LLM 只负责"怎么说"，不负责"等于几"。
"""
from __future__ import annotations

import logging
import os
import re
import threading

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr as _parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

log = logging.getLogger(__name__)

TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

# 单次 SymPy 计算的等待上限（秒）。simplify / limit 碰上病态表达式可能算很久，
# 不能让一个学生的答案把判定链路拖死。超时后放弃计算、判定退回下一层。
_SYMPY_TIMEOUT = float(os.environ.get("SYMPY_TIMEOUT", "5") or 5)

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

# ---------------------------------------------------------------- 语音口语 → 符号
#
# 语音输入转文字常写成"五倍x的四次方""3x方""负12"——直接进不了 SymPy，
# 判定只能要求学生重打一遍。这里做一轮**保守**的口语翻译：只翻有把握的模式
# （中文数字、N次方/平方/立方、N倍、负号），翻不动的原样放过，
# 交给下一层（关键词匹配 / 语义判定）——宁可少翻一层，不可瞎翻。

_CN_DIGIT_MAP = {"零": "0", "一": "1", "二": "2", "两": "2", "三": "3",
                 "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}
# 含"十"的组合先处理（十五→15、二十→20），再单字替换
_CN_TEN = re.compile(r"([零一二两三四五六七八九]?)十([零一二两三四五六七八九]?)")

_VOICE_POW = re.compile(r"([0-9xX)）])\s*的?\s*([0-9]+)\s*次[方幂]")     # x的4次方 → x^4
_VOICE_LIFANG = re.compile(r"([0-9xX)）])\s*的?\s*立方")                  # x立方 → x^3
_VOICE_PINGFANG = re.compile(r"([0-9xX)）])\s*的?\s*平方")                # x的平方 → x^2
# 语音里的"方"几乎都是"平方"的尾巴（3x方→3x^2）；但要避开"方程""方面"这些词
_VOICE_FANG = re.compile(r"([0-9xX)）])方(?![程面式向法便圆格针言阵块])")
_VOICE_BEI = re.compile(r"([0-9])\s*倍\s*之?")                            # 5倍x → 5*x
_VOICE_FU = re.compile(r"负(?=[0-9])")                                    # 负12 → -12
_NEG_CN = re.compile(r"负\s*([零一二两三四五六七八九十]+|[\d.]+)")          # 负十二 → -12


def has_cjk(s: str) -> bool:
    """这段话里有没有汉字——有的话多半是语音转文字的自然语言表述。"""
    return bool(_CJK.search(str(s or "")))


def _cn_to_digits(s: str) -> str:
    def ten(m):
        a = _CN_DIGIT_MAP.get(m.group(1), "1") if m.group(1) else "1"
        b = _CN_DIGIT_MAP.get(m.group(2), "0") if m.group(2) else ""
        return a + b
    s = _CN_TEN.sub(ten, s)
    for k, v in _CN_DIGIT_MAP.items():
        s = s.replace(k, v)
    return s.replace("零", "0")


def voice_to_expr(s: str) -> str:
    """把语音输入常见的中文数学口语翻成符号写法；不是口语的原样返回。"""
    s = str(s or "")
    if not _CJK.search(s) and "负" not in s:
        return s.translate(_FULL_DIGITS)      # 纯 ASCII 快速通道
    s = _cn_to_digits(s)
    s = _VOICE_POW.sub(r"\1^\2", s)
    s = _VOICE_LIFANG.sub(r"\1^3", s)
    s = _VOICE_PINGFANG.sub(r"\1^2", s)
    s = _VOICE_FANG.sub(r"\1^2", s)
    s = _VOICE_BEI.sub(r"\1*", s)
    s = _VOICE_FU.sub("-", s)
    return s.translate(_FULL_DIGITS)

# 受限命名空间：只放数学对象，不放 __import__ / open / eval 可达的东西
_EVAL_NS = {
    # 实数域符号：学校数学只在实数里做。默认的复数域 Symbol 会让
    # diff(Abs(x)**n, x) 化不开（带出一堆 re(x)/im(x)），判分必挂
    "x": sp.Symbol("x", real=True),
    "Symbol": sp.Symbol,
    "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "cot": sp.cot,
    "Abs": sp.Abs, "sign": sp.sign, "sgn": sp.sign,   # 导函数常用（|x|^k 的导数）
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


def _with_timeout(fn, args=(), timeout: float | None = None):
    """跑一段可能卡死的 SymPy 计算，超过 timeout 秒就不再等。

    Python 线程杀不掉，只能"不再等"：每次起一个守护线程，超时后把它留在
    后台自生自灭，主流程拿到 None。用完即弃（而不是共享线程池）是为了避免
    一个卡死的任务占住池位、让后续所有提交都立刻超时的连锁反应。
    """
    timeout = _SYMPY_TIMEOUT if timeout is None else timeout
    box: list = []

    def run():
        try:
            box.append(fn(*args))
        except Exception:
            pass

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    if box:
        return box[0]
    if t.is_alive():
        log.warning("SymPy %s 超过 %s 秒未返回，已放弃", getattr(fn, "__name__", fn), timeout)
    return None


def number_in(text: str):
    """从一段自然语言里抽出学生给出的最终数值。

    策略：分数/小数优先，整数只在没被分数吞掉时才算；取最靠右的一个
    （学生习惯把最终答案放在最后）。
    返回 sympy 表达式，抽不到返回 None。
    """
    text = (text or "").translate(_FULL_DIGITS)   # ２ → 2，否则 sympify 出来不是整数
    text = _FRAC_TEX.sub(r"\1/\2", text)          # \frac{1}{2} → 1/2

    # 中文分数词先查（"二分之一"不能被口语翻译拆成 2分之1）
    for word, val in _CN_WORDS:
        if word in text:
            return val

    # 语音口语的**窄**翻译：只处理"负 + 中文数字"（负十二 → -12）。
    # 不能在这里做整体中文数字转换——"二阶""三分之二"会被抽出错误的数字，
    # 反而把答对的学生判错。表达式级的口语翻译在 parse_student_expr 里做。
    text = _NEG_CN.sub(lambda m: "-" + _cn_to_digits(m.group(1)), text)

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


def _same_raw(a, b, tol: float) -> bool:
    diff = sp.simplify(sp.nsimplify(a) - sp.nsimplify(b))
    if diff == 0:
        return True
    if diff.is_number:
        return abs(complex(diff.evalf())) < tol
    return False


def _same_numeric_probe(a, b) -> bool:
    """符号化简证不出相等时，代几个有理点做数值抽查。

    Abs / sign 的不同写法（3*x**2*sign(Abs(x)) vs 3*x**2*sign(x)）simplify
    未必能证等，但代点必现形。全等函数在所有点上取值相同——抽查不等即不等；
    抽查全相等就当作相等（对伴学判定足够，也宁可宽容不错判）。
    """
    syms = list(getattr(a, "free_symbols", set()) | getattr(b, "free_symbols", set()))
    if not syms or len(syms) > 2:
        return False
    for point in (sp.Rational(7, 3), sp.Rational(-5, 2), sp.Rational(1, 7)):
        subs = {s: point for s in syms}
        try:
            va = complex(sp.N(a.subs(subs)))
            vb = complex(sp.N(b.subs(subs)))
        except Exception:
            return False            # 代不进去（奇点等）：保守判不等
        if abs(va - vb) > 1e-6 * max(1.0, abs(va), abs(vb)):
            return False
    return True


def same(a, b, tol: float = 1e-9) -> bool:
    """判断两个数学量是否相等：分数走符号化简，浮点走容差；
    含符号的表达式化简证不出时，再代点数值抽查（见 _same_numeric_probe）。"""
    if a is None or b is None:
        return False
    if _with_timeout(_same_raw, (a, b, tol)):
        return True
    return bool(_with_timeout(_same_numeric_probe, (a, b)))


# 学生会用来写答案的字符（"4x^3"、"y=9x-16"、"f(x)=x**2"）；出现别的就当不是表达式
_EXPR_ONLY = re.compile(r"^[0-9A-Za-z_+\-*/().,='^]+$")


def parse_student_expr(s: str):
    """把学生的整段输入当表达式解析（供答案本身是表达式——导函数等——的题用）。

    容忍：^ 幂号、缺乘号（4x^3）、带等号（y=9x-16 取右边）、全角符号。
    出现中文/奇怪的字符、或解析出的符号超出 x/t 就放弃（返回 None），
    由上层退回数字抽取。解析出的不是纯数值调用方也无需特判——same() 都能比。
    """
    s = str(s or "").strip()
    if not s or len(s) > 80:
        return None
    s = s.translate(_FULL)
    s = latex_to_ascii(s)            # 符号条插入的 LaTeX：x^{2} → x**2、\frac{1}{2} → 1/2
    s = voice_to_expr(s)             # 语音口语：3x方 → 3x^2、五倍x的四次方 → 5*x^4
    s = (s.replace("×", "*").replace("÷", "/").replace("－", "-")
          .replace("－", "-").replace("＊", "*"))
    if "=" in s:                     # "y=9x-16"、"f(x)=x^2" 取等号右边
        s = s.rsplit("=", 1)[-1].strip()
    if not _EXPR_ONLY.match(s):
        return None
    s = re.sub(r"\s+", "", s).replace("^", "**")
    if not s or _CJK.search(s) or _DANGEROUS.search(s):
        return None
    try:
        ns = {**_SAFE, "x": sp.Symbol("x", real=True), "t": sp.Symbol("t", real=True)}
        e = _parse_expr(s, local_dict=ns, transformations=TRANSFORMS)
    except Exception:
        return None
    if not isinstance(e, sp.Basic):
        return None
    free = {str(sym) for sym in getattr(e, "free_symbols", set())}
    if free - _ALLOWED_SYMBOLS:
        return None
    return e


def _limit_raw(expr_str: str, var: str, point):
    x = sp.Symbol(var)
    expr = _parse_expr(expr_str, local_dict={**_SAFE, var: x}, transformations=TRANSFORMS)
    return sp.limit(expr, x, point)


def limit_of(expr_str: str, var: str = "x", point=0):
    """符号求极限。失败或超时返回 None。"""
    return _with_timeout(_limit_raw, (expr_str, var, point))


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


def latex_to_ascii(s: str) -> str:
    """学生用符号条输进来的 LaTeX 片段 → SymPy 可解析的 ASCII。

    只做无损的结构转换：分式/根号/上标/花括号/常用命令。
    `\\lim`、`\\sum`、`\\int` 这类算子不是"答案表达式"，转不动就原样留下，
    交给语义层（模型翻译）处理——宁可少一层确定性，不可误判。
    """
    s = _TEX_SPACE.sub("", s)
    for _ in range(4):                       # 允许有限层嵌套
        new = _TEX_FRAC.sub(r"((\1)/(\2))", s)
        new = _TEX_SQRT.sub(r"sqrt(\1)", new)
        new = _TEX_POW.sub(r"**(\1)", new)
        if new == s:
            break
        s = new
    for cmd, rep in _TEX_CMD:
        s = s.replace(cmd, rep)
    return s.replace("\\", "").replace("{", "").replace("}", "")
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
    # zoo / nan / 无穷这类"求值出的怪东西"不是可判定的期望值
    # （实测模型把"分子分母各是几"的答案写成 0/0 → sympify 出 zoo，
    #  学生老老实实答 0 会被 same(0, zoo) 判成错）
    if expr in (sp.zoo, sp.nan, sp.oo, -sp.oo):
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


def _evaluate_raw(src: str):
    try:
        return eval(src, {"__builtins__": {}}, dict(_EVAL_NS))
    except Exception:
        return None


def evaluate(expr_str: str):
    """在受限命名空间里求值一个 SymPy 表达式，返回符号结果。

    四层防护：
      1. 字符串层面先挡掉 dunder 与动态执行关键字（`().__class__.__bases__[0]
         .__subclasses__()` 这种遍历在纯命名空间隔离下是能跑通的）
      2. 不给 `__builtins__`
      3. 命名空间里只有数学对象，没有任何 import 通道
      4. 病态表达式（深层 integrate / limit）可能算不完 —— 套超时，超时返回 None

    目前用于内置题库表达式与出题层预验证；若将来直接执行 LLM 生成的代码，
    还得加子进程隔离（大纲第三节第 3 条）。
    """
    src = str(expr_str)
    if _DANGEROUS.search(src):
        return None
    return _with_timeout(_evaluate_raw, (src,))


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
