# -*- coding: utf-8 -*-
"""判定层测试。

这一层是"学生随口一答会不会被冤枉"的唯一防线，所以测的是宽容度：
同一个意思的不同写法必须落到同一个判定上。
"""
from __future__ import annotations

import unittest

from app.core.judge import (
    BYPASS_PHRASES,
    hint_for,
    hits,
    normalize,
    verdict_offline,
)
from app.data.problems import PROBLEMS

LIMIT = PROBLEMS[0]["nodes"]


class TestNormalize(unittest.TestCase):
    def test_chinese_digit(self):
        self.assertEqual(normalize("二阶"), "2阶")
        self.assertEqual(normalize("两阶"), "2阶")

    def test_ordinal(self):
        self.assertEqual(normalize("第二阶"), "第2阶")

    def test_fullwidth(self):
        self.assertEqual(normalize("２阶"), "2阶")
        self.assertEqual(normalize("（２）"), "(2)")

    def test_superscript(self):
        self.assertEqual(normalize("x²"), "x^2")
        self.assertEqual(normalize("x⁵"), "x^5")

    def test_space_and_case(self):
        self.assertEqual(normalize(" X ^ 2 "), "x^2")
        self.assertEqual(normalize("Taylor"), "taylor")


class TestHits(unittest.TestCase):
    def test_partial_word(self):
        self.assertTrue(hits("用泰勒展开", ["泰勒", "麦克劳林"]))

    def test_case_insensitive(self):
        self.assertTrue(hits("taylor series", ["Taylor"]))

    def test_miss(self):
        self.assertFalse(hits("用洛必达", ["泰勒"]))

    def test_empty_words(self):
        self.assertFalse(hits("随便", None))


class TestOfflineVerdict(unittest.TestCase):
    """用极限题的四个节点做基准。"""

    def test_node1_substitution_gives_zero(self):
        for text in ("0", "０", "两个都是 0"):
            self.assertEqual(verdict_offline(text, LIMIT[0])[0], "correct", text)
        self.assertEqual(verdict_offline("1", LIMIT[0])[0], "wrong")

    def test_node2_undefined_form(self):
        for text in ("0/0", "0比0", "零比零", "是未定式"):
            self.assertEqual(verdict_offline(text, LIMIT[1])[0], "correct", text)

    def test_node2_method(self):
        self.assertEqual(verdict_offline("用泰勒展开", LIMIT[2])[0], "correct")
        self.assertEqual(verdict_offline("麦克劳林", LIMIT[2])[0], "correct")

    def test_node2_partial_when_detour_works(self):
        """洛必达走得通但不是本题最优——判 partial，不是错。"""
        self.assertEqual(verdict_offline("用洛必达", LIMIT[2])[0], "partial")

    def test_node3_order_accepts_many_phrasings(self):
        """这一条是上一轮返工的直接原因：答 2、二阶、x² 都得对。"""
        for text in ("2", "２", "二阶", "两阶", "第二阶", "x²", "x^2", "二次项"):
            self.assertEqual(verdict_offline(text, LIMIT[3])[0], "correct", text)

    def test_node3_wrong_order(self):
        self.assertEqual(verdict_offline("五阶", LIMIT[3])[0], "wrong")

    def test_node3_equivalence_detour(self):
        """等价无穷小在这题会失效，判 wrong 但带回应话术。"""
        verdict, _, reply = verdict_offline("等价无穷小", LIMIT[3])
        self.assertEqual(verdict, "wrong")
        self.assertIn("分母", reply)

    def test_node4_limit_value(self):
        for text in ("1/2", "0.5", "2/4", r"\frac{1}{2}", "二分之一", "一半"):
            self.assertEqual(verdict_offline(text, LIMIT[5])[0], "correct", text)

    def test_node4_wrong_value(self):
        self.assertEqual(verdict_offline("1/3", LIMIT[5])[0], "wrong")

    def test_gets_sympy_value_back(self):
        _, got, _ = verdict_offline("1/2", LIMIT[5])
        self.assertIsNotNone(got)


class TestAllProblems(unittest.TestCase):
    """题库里每个节点声明的标准答法，都必须被自己判对。"""

    def test_declared_answers_are_accepted(self):
        for prob in PROBLEMS:
            for node in prob["nodes"]:
                if node.get("numeric"):
                    self.assertEqual(
                        verdict_offline(str(node["numeric"]), node)[0], "correct",
                        f"{prob['id']} n{node['n']} numeric")
                if node.get("answers"):
                    self.assertEqual(
                        verdict_offline(node["answers"][0], node)[0], "correct",
                        f"{prob['id']} n{node['n']} answers")

    def test_garbage_is_not_correct(self):
        for prob in PROBLEMS:
            for node in prob["nodes"]:
                self.assertNotEqual(
                    verdict_offline("？？？", node)[0], "correct",
                    f"{prob['id']} n{node['n']}")

    def test_common_ways_of_saying_it_are_accepted(self):
        """学生用大白话/常见变体回答，不该被判错（匹配范围宁宽勿窄）。"""
        cases = [
            ("limit", 3, ["展开", "泰勒展开", "麦克劳林", "taylor", "泰勒公式"]),
            ("limit", 4, ["二阶", "2阶", "第二阶", "x的平方", "两阶", "二次"]),
            ("integral", 1, ["分部积分", "分步积分", "分布积分"]),
            ("integral", 2, ["e^x", "e的x次方", "exp(x)", "e^x+C"]),
        ]
        for prob_id, n, texts in cases:
            prob = next(p for p in PROBLEMS if p["id"] == prob_id)
            node = next(nd for nd in prob["nodes"] if nd["n"] == n)
            for t in texts:
                self.assertEqual(
                    verdict_offline(t, node)[0], "correct",
                    f"{prob_id} n{n} 应接受「{t}」")
        # 追问链路（分母换成 x^5 那一步）走自己的 answers
        fu = next(nd for nd in next(p for p in PROBLEMS if p["id"] == "limit")["nodes"]
                  if nd["n"] == 4)["followup"]
        for t in ["五阶", "5阶", "第五阶", "x^5", "5次"]:
            self.assertEqual(verdict_offline(t, fu)[0], "correct", f"追问应接受「{t}」")


class TestBypass(unittest.TestCase):
    def test_phrases(self):
        for text in ("给我答案", "别问了", "答案呢", "直接告诉我"):
            self.assertTrue(hits(text, BYPASS_PHRASES), text)

    def test_normal_answer_is_not_bypass(self):
        self.assertFalse(hits("我觉得是 1/2", BYPASS_PHRASES))


class TestHint(unittest.TestCase):
    def test_escalates_and_caps(self):
        node = LIMIT[0]
        self.assertEqual(hint_for(node, 0), node["hints"][0])
        self.assertEqual(hint_for(node, 99), node["hints"][-1])
        self.assertEqual(hint_for(node, -1), node["hints"][-1])

    def test_cloud_node_without_hints(self):
        self.assertEqual(hint_for({"question": "问什么？"}, 0), "问什么？")


class TestUnparseableExpect(unittest.TestCase):
    """模型给的 sympy检查 常常不是可求值表达式（"计算 y'=3x^2-3"）。

    这种字段必须当作"没有"，否则学生答对了数字也会被判错——错判比不判严重得多。
    """

    def test_garbage_numeric_falls_through_to_keywords(self):
        node = {"numeric": "计算 y' = 3x^2 - 3", "answers": ["9"]}
        self.assertEqual(verdict_offline("9", node)[0], "correct")

    def test_garbage_numeric_does_not_crash(self):
        node = {"numeric": "先求导再代入", "answers": ["9"]}
        self.assertEqual(verdict_offline("不会", node)[0], "wrong")


if __name__ == "__main__":
    unittest.main()
