# -*- coding: utf-8 -*-
"""教学引擎测试：完整闭环 + 每个硬约束。

覆盖大纲里"绝对不能违反"的几条：
  - 每条回复以问句结尾（学生看完必须知道该答什么）
  - 讨答案不推进节点
  - 连错三次才补答并推进
  - 标准答案永不下发给前端
  - 自评 → 变式 的顺序，用变式结果反过来裁决自评
"""
from __future__ import annotations

import json
import unittest
from unittest import mock

from app.core import engine as engine_mod
from app.core import llm
from app.core.engine import Engine, clean_ask, ensure_question, strip_retreat_prefix
from app.data.problems import PROBLEMS
from tests.support import expected_answer, force_mock, release_mock, walk

ALL_IDS = [p["id"] for p in PROBLEMS]


def finish_with_one_mistake() -> "object":
    """走完极限题，但在"展开阶数"这一步错一次——红灯预警的标准剧本。"""
    sess, _ = Engine.create(problem_id="limit")
    Engine.answer(sess, "0")
    Engine.answer(sess, "0/0")
    Engine.answer(sess, "泰勒")
    assert Engine.answer(sess, "五阶")["verdict"] == "wrong"
    walk(sess)
    return sess


class EngineCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        force_mock()

    @classmethod
    def tearDownClass(cls):
        release_mock()

    def assert_endswith_question(self, message: str, where: str = ""):
        self.assertTrue(
            message.rstrip().endswith(("?", "？")),
            f"{where} 回复没有以问句结尾：…{message[-40:]!r}",
        )


class TestCreate(EngineCase):
    def test_four_problems(self):
        self.assertEqual(len(ALL_IDS), 4)

    def test_payload_shape(self):
        sess, payload = Engine.create(problem_id="limit")
        for key in ("session_id", "mode", "problem", "meta", "total_nodes", "message", "stage", "progress"):
            self.assertIn(key, payload)
        self.assertEqual(payload["mode"], "mock")
        self.assertTrue(payload["offline_demo"])
        self.assertEqual(payload["progress"], [1, len(sess.nodes)])

    def test_unknown_id_falls_back_to_first(self):
        sess, payload = Engine.create(problem_id="不存在")
        self.assertEqual(payload["problem_id"], "limit")

    def test_standard_answer_never_sent(self):
        """护城河：学生拿到的 JSON 里不能出现标准答案字段。"""
        _, payload = Engine.create(problem_id="limit")
        blob = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn("_answer", blob)
        self.assertNotIn("标准答案", blob)


class TestWalkthrough(EngineCase):
    def test_every_problem_finishes(self):
        for pid in ALL_IDS:
            with self.subTest(problem=pid):
                sess, _ = Engine.create(problem_id=pid)
                steps = walk(sess)
                self.assertTrue(sess.finished, f"{pid} 没走完")
                self.assertEqual(len(sess.results), len(sess.nodes))
                last = steps[-1]
                self.assertEqual(last["next_action"], "self_assessment")

    def test_every_reply_ends_with_question(self):
        for pid in ALL_IDS:
            with self.subTest(problem=pid):
                sess, payload = Engine.create(problem_id=pid)
                self.assert_endswith_question(payload["message"], f"{pid} 首问")
                for i, step in enumerate(walk(sess)):
                    self.assert_endswith_question(step["message"], f"{pid} 第{i + 1}步")

    def test_hint_escalation_also_ends_with_question(self):
        """答错走的提示路径同样要收在问句上——这是上一轮返工的重点。"""
        sess, _ = Engine.create(problem_id="limit")
        for _ in range(2):
            step = Engine.answer(sess, "不知道")
            self.assertIn(step["verdict"], ("wrong", "partial"))
            self.assert_endswith_question(step["message"], "提示")


class TestGuardrails(EngineCase):
    def test_bypass_does_not_advance(self):
        sess, _ = Engine.create(problem_id="limit")
        for text in ("给我答案", "别问了", "答案呢"):
            step = Engine.answer(sess, text)
            self.assertEqual(step["verdict"], "bypass")
            self.assertEqual(sess.node_idx, 0)
            self.assertEqual(len(sess.results), 0)
            self.assertIn("不给", step["message"])

    def test_three_wrongs_reveal_and_advance(self):
        sess, _ = Engine.create(problem_id="limit")
        for _ in range(2):
            Engine.answer(sess, "不知道")
        self.assertEqual(sess.node_idx, 0, "两次答错不该推进")
        step = Engine.answer(sess, "还是不会")
        self.assertEqual(step["verdict"], "wrong")
        self.assertEqual(sess.node_idx, 1, "第三次答错应当补答并推进")
        self.assertEqual(sess.results[-1]["verdict"], "revealed")
        self.assertIn("我替你补上", step["message"])

    def test_empty_answer_treated_as_wrong(self):
        sess, _ = Engine.create(problem_id="limit")
        step = Engine.answer(sess, "   ")
        self.assertNotEqual(step["verdict"], "correct")
        self.assertEqual(sess.node_idx, 0)

    def test_followup_does_not_advance_immediately(self):
        """答对主问题后先追问，节点不推进——让学生自己说出规律。"""
        sess, _ = Engine.create(problem_id="limit")
        for text in ("0", "0/0", "泰勒"):
            Engine.answer(sess, text)
        step = Engine.answer(sess, "2")
        self.assertEqual(step["verdict"], "correct")
        self.assertEqual(sess.node_idx, 3, "追问阶段节点不应推进")
        self.assertTrue(sess.followup_done)
        step = Engine.answer(sess, "5")
        self.assertEqual(sess.node_idx, 4, "追问答完才推进")


class TestSelfAssessment(EngineCase):
    def _finish_with_one_mistake(self):
        return finish_with_one_mistake()

    def test_red_light_when_overconfident(self):
        sess = self._finish_with_one_mistake()
        warn = Engine.self_assess(sess, 5)
        self.assertEqual(warn["level"], "red")
        self.assertIn("虚假自信", warn["title"])
        self.assertIn("展开阶数", warn["reason"])

    def test_green_light_when_underconfident(self):
        sess, _ = Engine.create(problem_id="limit")
        walk(sess)
        warn = Engine.self_assess(sess, 1)
        self.assertEqual(warn["level"], "green")

    def test_neutral_when_consistent(self):
        sess, _ = Engine.create(problem_id="limit")
        walk(sess)
        self.assertEqual(Engine.self_assess(sess, 3)["level"], "neutral")

    def test_score_clamped(self):
        sess, _ = Engine.create(problem_id="limit")
        walk(sess)
        self.assertEqual(Engine.self_assess(sess, 999)["score"], 5)
        self.assertEqual(Engine.self_assess(sess, -5)["score"], 0)


class TestVariant(EngineCase):
    def _prepared(self, answer_text):
        sess = TestSelfAssessment._finish_with_one_mistake(self)
        Engine.self_assess(sess, 5)
        v = Engine.make_variant(sess)
        self.assertTrue(v["verified"], "变式题答案必须由 SymPy 算出")
        return sess, Engine.judge_variant(sess, answer_text)

    def test_correct_answer_clears_red_light(self):
        sess, res = self._prepared("1/6")
        self.assertTrue(res["correct"])
        self.assertTrue(res["cleared"])
        self.assertIn("自评是准的", res["conclusion"])

    def test_wrong_answer_aggravates(self):
        sess, res = self._prepared("1/3")
        self.assertFalse(res["correct"])
        self.assertIn("高估", res["conclusion"])

    def test_variant_before_self_assessment_defaults_to_2(self):
        """没自评就做变式时给中性基准，不能崩。"""
        sess, _ = Engine.create(problem_id="limit")
        walk(sess)
        Engine.make_variant(sess)
        res = Engine.judge_variant(sess, "1/6")
        self.assertEqual(res["self_score"], 2)


class TestReport(EngineCase):
    def test_report_structure(self):
        sess = finish_with_one_mistake()
        Engine.self_assess(sess, 5)
        Engine.make_variant(sess)
        Engine.judge_variant(sess, "1/6")
        rep = Engine.report(sess)

        self.assertEqual(rep["session_id"], sess.id)
        self.assertEqual(len(rep["bars"]), len(sess.nodes))
        for bar in rep["bars"]:
            self.assertIn("kp", bar)
            self.assertIn("score", bar)
            self.assertGreaterEqual(bar["score"], 0)
            self.assertLessEqual(bar["score"], 100)
        self.assertEqual(rep["虚假自信记录"]["状态"], "已解除（变式题独立答对）")
        self.assertEqual(rep["变式题"]["结果"], "正确")
        self.assertEqual(rep["总体掌握度"], round(sum(b["score"] for b in rep["bars"]) / len(rep["bars"])))

    def test_radar_aggregation(self):
        sess, _ = Engine.create(problem_id="limit")
        walk(sess)
        radar = Engine.radar(sess)
        self.assertEqual(len(radar["labels"]), len(sess.nodes))
        self.assertEqual(len(radar["values"]), len(radar["labels"]))
        self.assertTrue(radar["enough"], "四个知识点足够画雷达图")
        self.assertTrue(all(0 <= v <= 100 for v in radar["values"]))
        self.assertEqual(radar["answered"], len(sess.nodes))

    def test_radar_skips_unanswered(self):
        """没走到节点不能记 0，否则整张图会被压成一朵花。"""
        sess, _ = Engine.create(problem_id="limit")
        Engine.answer(sess, "0")
        radar = Engine.radar(sess)
        self.assertEqual(radar["answered"], 1)
        self.assertEqual(radar["labels"], ["直接代入"])
        self.assertFalse(radar["enough"], "不足 3 个维度前端应退回条形图")

    def test_partial_report_without_variant(self):
        sess, _ = Engine.create(problem_id="integral")
        Engine.answer(sess, "分部积分")
        rep = Engine.report(sess)
        self.assertIsNone(rep["变式题"])
        self.assertEqual(len(rep["bars"]), len(sess.nodes))
        self.assertEqual(rep["bars"][0]["score"], 100)
        self.assertEqual(rep["bars"][-1]["note"], "未涉及")


class TestDowngrade(EngineCase):
    """答错两次先退回到前置知识点，而不是直接给答案。"""

    def _at_limit_node3(self):
        sess, _ = Engine.create(problem_id="limit")
        for text in ("0", "0/0", "泰勒"):
            Engine.answer(sess, text)
        first = Engine.answer(sess, "五阶")
        self.assertEqual(first["verdict"], "wrong")
        return sess

    def test_second_wrong_falls_back_to_prerequisite(self):
        sess = self._at_limit_node3()
        step = Engine.answer(sess, "不知道")
        self.assertIn("先退一步", step["message"])
        self.assertTrue(sess.prereq_active)
        self.assertEqual(sess.node_idx, 3, "降维时不能推进节点")
        self.assert_endswith_question(step["message"], "降维提问")

    def test_correct_prerequisite_returns_to_main_question(self):
        sess = self._at_limit_node3()
        Engine.answer(sess, "不知道")
        step = Engine.answer(sess, "2")          # 前置题：分母最高次是 2
        self.assertFalse(sess.prereq_active)
        self.assertIn("回到刚才那一步", step["message"])
        self.assertEqual(sess.node_idx, 3, "回到主问题，节点仍未推进")

        step = Engine.answer(sess, "2")          # 主问题答对，但这一节点还有追问
        self.assertEqual(step["verdict"], "correct")
        self.assertEqual(sess.node_idx, 3)
        Engine.answer(sess, "5")                 # 追问答完才推进
        self.assertTrue(sess.results[-1]["downgraded"], "降维记录要写进作答记录")

    def test_wrong_prerequisite_gives_answer_and_returns(self):
        sess = self._at_limit_node3()
        Engine.answer(sess, "不知道")
        step = Engine.answer(sess, "五阶")        # 前置题答错
        self.assertIn("回到刚才那一步", step["message"])
        self.assertFalse(sess.prereq_active)
        self.assertEqual(sess.node_wrong, 2, "降维不算新的答错")

    def test_third_wrong_still_reveals(self):
        sess = self._at_limit_node3()
        Engine.answer(sess, "不知道")             # 触发降维
        Engine.answer(sess, "五阶")               # 前置题答错，回到主问题
        step = Engine.answer(sess, "还是不会")     # 第三次错
        self.assertEqual(step["verdict"], "wrong")
        self.assertEqual(sess.node_idx, 4, "第三次答错才补答并推进")
        self.assertEqual(sess.results[-1]["verdict"], "revealed")

    def test_downgrade_only_once_per_node(self):
        sess = self._at_limit_node3()
        Engine.answer(sess, "不知道")
        self.assertTrue(sess.prereq_used)
        Engine.answer(sess, "2")                  # 前置答对，回到主问题
        step = Engine.answer(sess, "不知道")       # 主问题再错
        self.assertNotIn("先退一步", step["message"], "同一个节点不该反复降维")

    def test_report_marks_downgraded_kp(self):
        sess = self._at_limit_node3()
        Engine.answer(sess, "不知道")
        Engine.answer(sess, "2")
        walk(sess)
        bars = {b["kp"]: b for b in Engine.report(sess)["bars"]}
        self.assertIn("降维", bars["展开阶数"]["note"])
        self.assertLess(bars["展开阶数"]["score"], 100)


class TestCloudPlan(EngineCase):
    """模型拆题的输出清洗。

    实测踩到：Gemma 4 把「教学意图」写成"让学生掌握求导的规则"，
    学生对着一句教师备注没法回答。字段质量模型说了算，出口得兜住。
    """

    def _plan(self, payload, mode="cloud"):
        with mock.patch.object(engine_mod.llm, "chat_json", return_value=payload), \
             mock.patch.object(engine_mod.llm, "mode", return_value=mode):
            return engine_mod.cloud_plan("求切线")

    def test_teacher_talk_is_stripped(self):
        p = self._plan({"节点计划": [{"提问": "让学生掌握求导的规则，为后续做准备。",
                                      "知识点": "求导"}]})
        q = p["nodes"][0]["question"]
        self.assertNotIn("让学生", q)
        self.assertTrue(q.endswith(("?", "？")), q)

    def test_real_question_passes_through(self):
        p = self._plan({"节点计划": [{"提问": "求导之后得到什么？", "知识点": "求导"}]})
        self.assertEqual(p["nodes"][0]["question"], "求导之后得到什么？")

    def test_legacy_intent_field_still_works(self):
        p = self._plan({"节点计划": [{"教学意图": "让学生求导", "知识点": "求导"}]})
        self.assertIn("求导", p["nodes"][0]["question"])

    def test_local_budget_is_smaller(self):
        payload = {"节点计划": [{"提问": f"第 {i} 问？"} for i in range(1, 9)]}
        self.assertEqual(len(self._plan(payload, mode="local")["nodes"]), 3)
        self.assertEqual(len(self._plan(payload, mode="cloud")["nodes"]), 6)

    def test_clean_ask_handles_prefixes(self):
        for raw in ("这一步要让学生想清楚什么", "引导学生求导", "要求学生算出斜率"):
            self.assertTrue(len(clean_ask(raw)) < len(raw), raw)

    def test_retreat_prefix_not_duplicated(self):
        """降维问句自带"先退一小步"时，不能和引擎的"先退一步"叠成复读。"""
        self.assertEqual(strip_retreat_prefix("先退一小步：$e^{0}$ 等于几？"),
                         "$e^{0}$ 等于几？")
        self.assertEqual(strip_retreat_prefix("先退一步：想一下"), "想一下")
        self.assertEqual(strip_retreat_prefix("先只看分母：最高次是几？"),
                         "先只看分母：最高次是几？")   # 内容性前缀保留

    def test_short_field_format(self):
        """新格式用「问/答/式」三个短字段——本地模型漏 key 的概率低很多。"""
        p = self._plan({"节点": [{"问": "求导之后得到什么？", "答": "9", "式": "9"}]})
        n = p["nodes"][0]
        self.assertEqual(n["question"], "求导之后得到什么？")
        self.assertEqual(n["numeric"], "9")
        self.assertEqual(n["answers"], ["9"])

    def test_numeric_is_cleaned_from_latex(self):
        """模型写 LaTeX 也要能进 SymPy。"""
        p = self._plan({"节点": [{"问": "极限是几？", "答": "1/2", "式": r"$\frac{1}{2}$"}]})
        self.assertEqual(p["nodes"][0]["numeric"], "1/2")

    def test_numeric_falls_back_to_answer_field(self):
        """「式」写成一句人话，就退回「答」里那个干净的数。"""
        p = self._plan({"节点": [{"问": "极限是几？", "答": "1/2", "式": "约等于 0.5"}]})
        self.assertEqual(p["nodes"][0]["numeric"], "1/2")

    def test_numeric_is_empty_when_nothing_parses(self):
        """洗不出数值就留空：判定退回关键词匹配，绝不让 SymPy 瞎判。"""
        p = self._plan({"节点": [{"问": "用哪种方法？", "答": "等价无穷小", "式": "二阶"}]})
        self.assertEqual(p["nodes"][0]["numeric"], "")

    def test_broken_numeric_never_wrong_judges(self):
        """模型把「式」写坏时，答对的学生也绝不能被判错。"""
        from app.core.judge import verdict_offline
        node = {"numeric": "", "answers": ["0"], "partial_kw": [], "branches": []}
        verdict, _, _ = verdict_offline("0", node)
        self.assertEqual(verdict, "correct")


class TestSalvage(EngineCase):
    """JSON 结构坏了（漏 key、少括号），只要字段名还在就能把教学流程救回来。

    实测 Gemma 4 写出过 {"n":1,"提问":"...","分部积分的变量选择","e^x"}——
    整段 JSON 失效，但问句本身是好的。丢掉它等于白等一次推理。
    """

    BROKEN = ('{"节点":[{"n":1,"提问":"求导得到什么？","知识点":"求导"},'
              '{"n":2,"提问":"代入 $x=2$ 得几？"')

    def test_extract_questions_from_broken_json(self):
        qs = engine_mod.extract_questions(self.BROKEN)
        self.assertEqual(len(qs), 2)
        self.assertEqual(qs[0]["提问"], "求导得到什么？")

    def test_extract_keeps_latex_backslash(self):
        qs = engine_mod.extract_questions('{"提问":"$\\int x\\,dx$ 等于几？"}')
        self.assertIn("\\int", qs[0]["提问"])

    def test_plan_falls_back_to_salvage(self):
        with mock.patch.object(engine_mod.llm, "chat_json",
                               side_effect=llm.LLMError("JSON 解析连续失败：无法解析")), \
             mock.patch.object(engine_mod.llm, "last_reply", return_value=self.BROKEN):
            p = engine_mod.cloud_plan("求切线")
        self.assertEqual(len(p["nodes"]), 2)
        self.assertIn("求导", p["nodes"][0]["question"])

    def test_non_json_error_is_not_swallowed(self):
        """Key 错了要报 Key 错的，不能伪装成"拆题成功但节点为空"。"""
        with mock.patch.object(engine_mod.llm, "chat_json",
                               side_effect=llm.LLMError("HTTP 401 · invalid key")):
            with self.assertRaises(llm.LLMError):
                engine_mod.cloud_plan("求切线")

    def test_empty_salvage_still_raises(self):
        with mock.patch.object(engine_mod.llm, "chat_json",
                               side_effect=llm.LLMError("JSON 解析连续失败")), \
             mock.patch.object(engine_mod.llm, "last_reply", return_value="完全不是 JSON"):
            with self.assertRaises(llm.LLMError):
                engine_mod.cloud_plan("求切线")


class TestChoices(EngineCase):
    """不方便手机输入的地方给选项；选项必须与当前这一问对得上。"""

    def test_first_node_has_choices(self):
        _, payload = Engine.create(problem_id="limit")
        self.assertEqual(len(payload["choices"]), 3)
        self.assertEqual(payload["choices"][0], {"label": "$0$", "value": "0"})

    def test_choices_follow_the_next_question(self):
        sess, _ = Engine.create(problem_id="limit")
        step = Engine.answer(sess, "0")
        values = [c["value"] for c in step["choices"]]
        self.assertIn("0/0型", values, "答完第一问，第二问的选项要跟着换")

    def test_tapping_a_choice_is_judged_correct(self):
        """点选项等于输入 value，判定层不需要知道有选择题这回事。"""
        sess, _ = Engine.create(problem_id="limit")
        first = sess.nodes[0]["choices"][0]["value"]
        self.assertEqual(Engine.answer(sess, first)["verdict"], "correct")

    def test_nodes_without_choices_return_empty(self):
        sess, _ = Engine.create(problem_id="tangent")
        _, payload = Engine.create(problem_id="tangent")
        self.assertEqual(payload["choices"], [])

    def test_choices_cleared_when_finished(self):
        sess, _ = Engine.create(problem_id="limit")
        walk(sess)
        step = Engine.answer(sess, "1/2")
        self.assertEqual(step.get("choices", []), [])

    def test_finished_reply_still_has_progress(self):
        """课后还继续说话时也要带 progress。

        前端拿它去画 1/6 的进度点，少一个字段就是一条
        "Cannot read properties of undefined" 的红条（实机截到过）。
        """
        sess, _ = Engine.create(problem_id="limit")
        walk(sess)
        self.assertTrue(sess.finished)
        step = Engine.answer(sess, "1/2")
        self.assertEqual(step["progress"], [len(sess.nodes), len(sess.nodes)])


class TestNoSympyNote(EngineCase):
    """每轮的 SymPy 核对提示已撤掉：对了是废话，错了等于报答案。"""

    def test_correct_answer_has_no_note(self):
        sess, _ = Engine.create(problem_id="limit")
        step = Engine.answer(sess, "0")
        self.assertEqual(step["verdict"], "correct")
        self.assertEqual(step["note"], "")

    def test_wrong_answer_does_not_leak_answer(self):
        sess, _ = Engine.create(problem_id="limit")
        Engine.answer(sess, "0")
        step = Engine.answer(sess, "1")      # 第二步答错
        self.assertEqual(step["note"], "")
        self.assertNotIn("正确答案", step["message"])


class TestQuestionInvariant(EngineCase):
    """大纲硬规则第 3 条：每条回复必须以问句结尾。数据、模型都可能漏，所以有出口保险。"""

    def test_ensure_question_passthrough(self):
        self.assertEqual(ensure_question("这题怎么做？"), "这题怎么做？")
        self.assertEqual(ensure_question("这题怎么做?  "), "这题怎么做?")

    def test_ensure_question_appends(self):
        out = ensure_question("这是陈述句。")
        self.assertTrue(out.endswith("？"))
        self.assertIn("这是陈述句。", out)

    def test_ensure_question_handles_empty(self):
        self.assertEqual(ensure_question(""), "再想一想？")

    def test_every_hint_level_ends_with_question(self):
        """一路答错走完四道题，逐条检查——这条覆盖每级提示和"补答"路径。"""
        for pid in ALL_IDS:
            with self.subTest(problem=pid):
                sess, _ = Engine.create(problem_id=pid)
                for i in range(40):
                    if sess.finished:
                        break
                    step = Engine.answer(sess, "不知道")
                    self.assert_endswith_question(step["message"], f"{pid} 第{i + 1}次答错")
                self.assertTrue(sess.finished, f"{pid} 一路答错也应该能走完")


if __name__ == "__main__":
    unittest.main()
