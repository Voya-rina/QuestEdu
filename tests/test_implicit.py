# -*- coding: utf-8 -*-
"""隐函数求导：真值层、拆解校验、兜底骨架。

这道题（xy²−ln(x+1)+ln y=1，求 x=0 处的 dy/dx）是提示词里的示例，也是
小模型实测最容易出错的一类：把 xy² 的导数写成 2xy+y²y'、把 ln(x+1) 的负号
吃掉、把答案写成 1−e⁻¹。这些错误靠提示词压不住，只能靠符号引擎兜。
"""
from __future__ import annotations

import unittest

import sympy as sp

from app.core import engine, mathcheck

PROBLEM = "设 y=y(x) 由方程 xy^2 - ln(x+1) + ln y = 1 确定，求 x=0 处的 dy/dx。"


def truth_of(problem: str = PROBLEM) -> dict:
    t = mathcheck.implicit_derivative(problem)
    assert t is not None, "真值层没认出这道题"
    return t


def node(n: int, question: str, expect: str, numeric: str = "") -> dict:
    return {"n": n, "stage": "回顾", "knowledge": "链式法则", "question": question,
            "expect": expect, "numeric": numeric, "answers": [expect],
            "partial_kw": [], "correct_reply": "", "partial_hint": "", "hints": []}


class TestTruth(unittest.TestCase):
    """符号引擎算出来的真相。"""

    def test_y_at_point(self):
        self.assertTrue(mathcheck.same(truth_of()["y0"], sp.E))

    def test_final_value(self):
        self.assertTrue(mathcheck.same(truth_of()["value"], sp.E - sp.E ** 3))

    def test_other_right_hand_side(self):
        """右边换成 2：y(0)=e²，值 e²−e⁶。答案必须跟着题面走，不能抄示例。"""
        t = truth_of("已知 xy^2 - ln(x+1) + ln y = 2，求 x=0 处的 dy/dx。")
        self.assertTrue(mathcheck.same(t["y0"], sp.E ** 2))
        self.assertTrue(mathcheck.same(t["value"], sp.E ** 2 - sp.E ** 6))

    def test_not_implicit(self):
        self.assertIsNone(mathcheck.implicit_derivative("求 lim(x->0) (e^x-1-x)/x^2"))

    def test_latex_problem(self):
        """模型出的变式题是 LaTeX 写法，花括号会把方程截断成半截式子。"""
        t = truth_of("已知 $xy^{2}-\\ln(x+1)+\\ln y=5$（$y=y(x)$），"
                     "求 $\\left.\\frac{dy}{dx}\\right|_{x=0}$")
        self.assertTrue(mathcheck.same(t["y0"], sp.E ** 5))
        self.assertTrue(mathcheck.same(t["value"], sp.E ** 5 - sp.E ** 15))


class TestChecks(unittest.TestCase):
    """每一步的数学能不能被认出来——正确的不许误判，错误的不许放过。"""

    def setUp(self):
        self.t = truth_of()

    def test_derivative_equation(self):
        for text in ("y²+2xyy'-1/(x+1)+y'/y=0",
                     "y'(2xy+1/y) = 1/(x+1) - y^2",
                     "y^2+2xyy'-1/(x+1)+y'/y"):
            self.assertTrue(mathcheck.equation_diff_ok(text, self.t), text)

    def test_wrong_derivative_equation(self):
        """实测错误：xy² 的导数写成 2xy+y²y'。"""
        self.assertFalse(mathcheck.equation_diff_ok(
            "2xy + y^2·y' - 1/(x+1) + y'/y = 0", self.t))

    def test_dy_expression(self):
        self.assertTrue(mathcheck.dy_expr_ok("y'=(1/(x+1)−y²)/(2xy+1/y)", self.t))
        self.assertTrue(mathcheck.dy_expr_ok("(1/(x+1)-y^2)/(2xy+1/y)", self.t))
        self.assertFalse(mathcheck.dy_expr_ok("y'=(1/(x+1)-2xy-y/x)/y'", self.t))

    def test_term_derivatives(self):
        self.assertTrue(mathcheck.term_deriv_ok("y^2+2xyy'", self.t))
        self.assertTrue(mathcheck.term_deriv_ok("y'/y", self.t))
        self.assertFalse(mathcheck.term_deriv_ok("2xy + y^2*y'", self.t))

    def test_term_derivative_sign(self):
        """-ln(x+1) 的导数是 -1/(x+1)；吃掉负号要被抓住。"""
        self.assertTrue(mathcheck.x_term_deriv_ok("-1/(x+1)", self.t))
        self.assertFalse(mathcheck.x_term_deriv_ok("1/(x+1)", self.t))

    def test_answer_is_original_equation(self):
        self.assertTrue(mathcheck.same_equation_ok("xy^2 - ln(x+1) + ln y = 1", self.t))
        self.assertFalse(mathcheck.same_equation_ok("y^2+2xyy'-1/(x+1)+y'/y=0", self.t))

    def test_superscript_value(self):
        self.assertTrue(mathcheck.same(
            mathcheck.math_text_to_expr("e-e³"), sp.E - sp.E ** 3))


class TestBuildNodes(unittest.TestCase):
    """整条方程当「答」时不能留下数值答案。"""

    def test_equation_answer_clears_numeric(self):
        nodes = engine._build_nodes(
            [{"问": "求导后的等式是什么？", "答": "y²+2xyy'-1/(x+1)+y'/y=0", "式": ""}],
            {"知识点": ["链式函数"]}, 10)
        self.assertEqual(nodes[0]["numeric"], "")

    def test_derivative_label_keeps_numeric(self):
        """`y'=3x^2-3` 这种右式就是答案，不能被当成方程扔掉。"""
        nodes = engine._build_nodes(
            [{"问": "导函数是什么？", "答": "y'=3x^2-3", "式": "3*x^2-3"}],
            {"知识点": ["求导"]}, 10)
        self.assertEqual(nodes[0]["numeric"], "3*x**2 - 3")


class TestPlanIssues(unittest.TestCase):
    """拆解校验：能过的好计划，和必须打回的三类硬伤。"""

    def setUp(self):
        self.t = truth_of()

    def good_plan(self) -> list[dict]:
        return [
            node(1, "方程两边对 x 求导，求导后的等式是什么？",
                 "y^2+2xyy'-1/(x+1)+y'/y=0"),
            node(2, "解出 y'，y' 等于多少？", "y'=(1/(x+1)-y^2)/(2xy+1/y)"),
            node(3, "把 x=0 代回原方程，y 等于几？", "e", "E"),
            node(4, "把 x=0、y=e 代进 y' 的表达式，dy/dx 等于几？",
                 "e-e^3", "E - exp(3)"),
        ]

    def test_good_plan_passes(self):
        self.assertEqual(engine.plan_issues(self.good_plan(), self.t), [])

    def test_per_term_plan_passes(self):
        """逐项求导也是对的教法，不能因为没写完整等式就否掉。"""
        plan = [
            node(1, "xy^2 对 x 求导等于多少？", "y^2+2xyy'"),
            node(2, "-ln(x+1) 对 x 求导等于多少？", "-1/(x+1)"),
            node(3, "ln y 对 x 求导等于多少？", "y'/y"),
            node(4, "解出 y'，y' 等于多少？", "y'=(1/(x+1)-y^2)/(2xy+1/y)"),
            node(5, "把 x=0 代回原方程，y 等于几？", "E", "E"),
            node(6, "dy/dx 等于几？", "E-e**3", "E - exp(3)"),
        ]
        self.assertEqual(engine.plan_issues(plan, self.t), [])

    def test_wrong_derivative_rejected(self):
        plan = self.good_plan()
        plan[0]["expect"] = "2xy + y^2*y' - 1/(x+1) + y'/y = 0"
        self.assertTrue(engine.plan_issues(plan, self.t))

    def test_wrong_final_value_rejected(self):
        plan = self.good_plan()
        plan[-1].update(expect="1-e^-1", numeric="1 - exp(-1)")
        self.assertTrue(engine.plan_issues(plan, self.t))

    def test_missing_y_node_rejected(self):
        plan = self.good_plan()
        del plan[2]
        for i, n in enumerate(plan, start=1):
            n["n"] = i
        self.assertTrue(engine.plan_issues(plan, self.t))

    def test_leaking_answer_rejected(self):
        """问句里写出"得 y=e"，学生就没得算了。"""
        plan = self.good_plan()
        plan[2]["question"] = "把 x=0 代入原方程：0·y^2-ln(1)+ln y=1，得 y=e，结果是多少？"
        self.assertTrue(engine.plan_issues(plan, self.t))

    def test_two_questions_in_one_node_rejected(self):
        plan = self.good_plan()
        plan[0]["question"] = "方程两边对 x 求导得什么？xy^2 的导数又是多少？"
        self.assertTrue(engine.plan_issues(plan, self.t))

    def test_self_doubting_answer_rejected(self):
        """实测：模型把「答」写成 "e^2-0+1=2≠1，此处需重新检查：…" 这种独白。"""
        plan = self.good_plan()
        plan[2]["expect"] = "e^2-0+1=2≠1，此处需重新检查：实际上 x=0 时得 y=e"
        plan[2]["numeric"] = ""
        self.assertTrue(engine.plan_issues(plan, self.t))

    def test_lowercase_leak_rejected(self):
        """模型把 y 的值写成小写 e 也一样是泄题。"""
        plan = self.good_plan()
        plan[2]["question"] = "把点 (0, e) 代入原方程验证，y 值是否正确？"
        self.assertTrue(engine.plan_issues(plan, self.t))

    def test_empty_plan_does_not_crash(self):
        self.assertTrue(engine.plan_issues([], self.t))


class TestAffirm(unittest.TestCase):
    """判对之后那句话：自相矛盾的回复要换掉。"""

    def test_keeps_short_affirmation(self):
        self.assertEqual(engine._affirm("对，就是这样。", {"correct_reply": ""}),
                         "对，就是这样。")

    def test_replaces_contradictory_reply(self):
        bad = "请写出原方程的具体形式，以便我们将 x=0 代入求解。"
        self.assertEqual(engine._affirm(bad, {"correct_reply": ""}), "对，就是这样。")

    def test_replaces_question(self):
        self.assertEqual(engine._affirm("那你能说说为什么吗？", {"correct_reply": "很好"}),
                         "很好")


class TestSkeleton(unittest.TestCase):
    """模型彻底拆不对时的兜底：可以朴素，不能教错。"""

    def setUp(self):
        self.t = truth_of()
        self.nodes = engine._implicit_skeleton(self.t)

    def test_complete(self):
        self.assertEqual(len(self.nodes), 4)
        for n in self.nodes:
            self.assertTrue(n["question"].endswith(("？", "?")), n["question"])
            self.assertTrue(n["expect"].strip())
            self.assertFalse(engine._node_math(n["expect"], self.t) == "bad", n["expect"])

    def test_ends_at_the_asked_quantity(self):
        last = self.nodes[-1]
        self.assertIn("dy/dx", last["question"])
        self.assertTrue(mathcheck.same(mathcheck.parse_expr(last["numeric"]),
                                       self.t["value"]))

    def test_passes_its_own_checks(self):
        self.assertEqual(engine.plan_issues(self.nodes, self.t), [])


if __name__ == "__main__":
    unittest.main()
