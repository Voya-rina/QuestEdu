# -*- coding: utf-8 -*-
"""确定性计算层测试：所有数字的唯一来源，必须绝对可靠。"""
from __future__ import annotations

import unittest

import sympy as sp

from app.core import mathcheck


class TestNumberExtraction(unittest.TestCase):
    """学生随口一答，能不能正确抽出数值。"""

    def test_fraction(self):
        self.assertTrue(mathcheck.same(mathcheck.number_in("所以极限是 1/2"), sp.Rational(1, 2)))

    def test_decimal(self):
        self.assertTrue(mathcheck.same(mathcheck.number_in("答案是 0.5"), sp.Rational(1, 2)))

    def test_zero(self):
        self.assertEqual(mathcheck.number_in("答案是0"), 0)

    def test_equivalent_fraction(self):
        self.assertTrue(mathcheck.same(mathcheck.number_in("2/4"), sp.Rational(1, 2)))

    def test_latex_frac(self):
        self.assertTrue(mathcheck.same(mathcheck.number_in(r"\frac{1}{2}"), sp.Rational(1, 2)))

    def test_fullwidth_digit(self):
        """全角数字曾让 sympify 解析出非整数，必须转半角。"""
        self.assertEqual(mathcheck.number_in("２"), 2)

    def test_negative(self):
        self.assertEqual(mathcheck.number_in("k=-16"), -16)

    def test_chinese_word(self):
        self.assertTrue(mathcheck.same(mathcheck.number_in("二分之一"), sp.Rational(1, 2)))

    def test_ignores_exponent(self):
        """x² 里的 2 不是答案。"""
        self.assertIsNone(mathcheck.number_in("x^2"))
        self.assertIsNone(mathcheck.number_in("二阶"))

    def test_takes_last_number(self):
        """学生习惯把最终结果放最后。"""
        self.assertEqual(mathcheck.number_in("先算 3 再减 1，最后是 2"), 2)

    def test_empty(self):
        self.assertIsNone(mathcheck.number_in(""))


class TestComparison(unittest.TestCase):
    def test_rational_equivalence(self):
        self.assertTrue(mathcheck.same(sp.Rational(1, 2), sp.Rational(2, 4)))

    def test_float_tolerance(self):
        self.assertTrue(mathcheck.same(0.5, sp.Rational(1, 2)))

    def test_not_equal(self):
        self.assertFalse(mathcheck.same(sp.Rational(1, 3), sp.Rational(1, 2)))

    def test_none_safe(self):
        self.assertFalse(mathcheck.same(None, sp.Rational(1, 2)))


class TestSandbox(unittest.TestCase):
    def test_limit(self):
        got = mathcheck.evaluate("limit((exp(x) - 1 - x)/x**2, x, 0)")
        self.assertTrue(mathcheck.same(got, sp.Rational(1, 2)))

    def test_integral(self):
        got = mathcheck.evaluate("integrate(x*exp(x), (x, 0, 1))")
        self.assertTrue(mathcheck.same(got, 1))

    def test_matrix_eigen(self):
        got = mathcheck.evaluate("max(Matrix([[2, 1], [1, 2]]).eigenvals().keys())")
        self.assertTrue(mathcheck.same(got, 3))

    def test_blocks_import(self):
        """受限命名空间：没有任何 import 通道。"""
        self.assertIsNone(mathcheck.evaluate("__import__('os').system('echo hi')"))

    def test_blocks_dunder(self):
        self.assertIsNone(mathcheck.evaluate("().__class__.__bases__"))

    def test_bad_expression_returns_none(self):
        self.assertIsNone(mathcheck.evaluate("1/0"))


class TestCleanNumeric(unittest.TestCase):
    """约束模型输出：节点的「期望值」必须先洗成可求值的表达式。

    模型写什么都可能。最坏的不是解析抛异常，而是 `sympify("二阶")`
    **成功**返回一个叫「二阶」的符号——判定从此变成掷骰子。
    """

    def test_plain_numbers(self):
        for raw, want in [("0", "0"), ("1/2", "1/2"), ("-16", "-16"),
                          ("0.5", "1/2"), ("Rational(1,2)", "1/2")]:
            self.assertEqual(mathcheck.clean_numeric(raw), want, raw)

    def test_latex_is_converted(self):
        self.assertEqual(mathcheck.clean_numeric(r"$\frac{1}{2}$"), "1/2")
        self.assertEqual(mathcheck.clean_numeric(r"\frac{1}{2}"), "1/2")
        self.assertEqual(mathcheck.clean_numeric(r"\left(1+x\right)^{2}"), "(x + 1)**2")

    def test_implicit_multiplication(self):
        self.assertEqual(mathcheck.clean_numeric("3x^2-3"), "3*x**2 - 3")
        self.assertEqual(mathcheck.clean_numeric("2pi"), "2*pi")

    def test_equation_takes_right_hand_side(self):
        self.assertEqual(mathcheck.clean_numeric("y'=3x^2-3"), "3*x**2 - 3")
        self.assertEqual(mathcheck.clean_numeric("x=2"), "2")

    def test_prose_is_rejected(self):
        for raw in ("二阶", "约等于 0.5", "x = 2 时", "答案是 1/2", "", "   "):
            self.assertEqual(mathcheck.clean_numeric(raw), "", raw)

    def test_unit_symbol_is_rejected(self):
        """`2cm` 会被 sympify 成一个叫 cm 的符号，不能当 2 用。"""
        self.assertEqual(mathcheck.clean_numeric("2cm"), "")

    def test_superscript_chars_are_rejected(self):
        """3x²-3 里那个 ² 不是 ASCII，宁可留空也不要猜。"""
        self.assertEqual(mathcheck.clean_numeric("3x²-3"), "")

    def test_dangerous_input_is_rejected(self):
        self.assertEqual(mathcheck.clean_numeric("__import__('os')"), "")

    def test_result_is_always_parseable(self):
        """洗完的字符串一定能被 parse_expr 解析（判定层的唯一前提）。"""
        for raw in ("0", "1/2", "0.5", r"\frac{1}{2}", "3x^2-3", "sqrt(2)/2", "e^2"):
            cleaned = mathcheck.clean_numeric(raw)
            self.assertTrue(cleaned, raw)
            self.assertIsNotNone(mathcheck.parse_expr(cleaned), raw)


class TestStudentExpr(unittest.TestCase):
    """答案本身是表达式（导函数等）时，把学生输入整体解析后符号比对。"""

    def test_parse_tolerant_forms(self):
        x = sp.Symbol("x")
        cases = {
            "4x^3": 4 * x**3,          # 缺乘号 + ^ 幂号
            "4x**3": 4 * x**3,
            "4*x**3": 4 * x**3,
            "y=9x-16": 9 * x - 16,      # 带等号取右边
            "f(x)=x^2": x**2,
        }
        for raw, want in cases.items():
            got = mathcheck.parse_student_expr(raw)
            self.assertIsNotNone(got, raw)
            self.assertTrue(mathcheck.same(got, want), f"{raw} → {got}")

    def test_reject_non_expr(self):
        for raw in ("", "二阶", "我不会", "f'(x)=?", "__import__('os')",
                    "say hello world"):
            self.assertIsNone(mathcheck.parse_student_expr(raw), raw)


class TestSameProbe(unittest.TestCase):
    """same() 的代点抽查：Abs/sign 不同写法 simplify 证不出，代点必现形。"""

    def test_equivalent_sign_forms(self):
        x = sp.Symbol("x")
        # |x|^4 的导数：sympy 给 4*Abs(x)**3*sign(x)，学生写 4x^3——代点等值
        self.assertTrue(mathcheck.same(
            sp.sympify("4*Abs(x)**3*sign(x)"), sp.sympify("4*x**3")))
        # |x|^3 的导数：3x|x| 与 3x^2*sign(x) 是同一个函数
        self.assertTrue(mathcheck.same(
            sp.sympify("3*x*Abs(x)"), sp.sympify("3*x**2*sign(x)")))
        # 反例：sign(Abs(x)) 与 sign(x) 不是同一个函数（x<0 时一个 +1 一个 -1），
        # probe 必须能区分，不能把它们误判为等价
        self.assertFalse(mathcheck.same(
            sp.sympify("3*x**2*sign(Abs(x))"), sp.sympify("3*x**2*sign(x)")))

    def test_different_exprs_stay_false(self):
        x = sp.Symbol("x")
        self.assertFalse(mathcheck.same(sp.sympify("x"), sp.sympify("x + 1")))
        self.assertFalse(mathcheck.same(sp.sympify("x**2"), sp.sympify("x**3")))
        self.assertFalse(mathcheck.same(sp.Integer(3), sp.Integer(4)))


if __name__ == "__main__":
    unittest.main()
