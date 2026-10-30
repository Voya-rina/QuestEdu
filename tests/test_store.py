# -*- coding: utf-8 -*-
"""持久化测试：五张表是否真的建了、会话断了能不能接上。

最重要的一条是 `test_resume_after_restart`：清掉内存缓存（模拟服务重启）后，
会话必须还能继续答——现在重启就 404，这是答辩现场最容易被问倒的一点。
"""
from __future__ import annotations

import unittest

from sqlalchemy import inspect, select

from app.core.engine import Engine
from app.core.store import SqlStore, store
from app.db import engine
from tests.support import force_mock, release_mock, walk


class StoreCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        force_mock()

    @classmethod
    def tearDownClass(cls):
        release_mock()

    def setUp(self):
        self.assertIsInstance(store, SqlStore, "测试应跑在 SQLite 上")


class TestSchema(StoreCase):
    def test_five_tables_exist(self):
        tables = set(inspect(engine).get_table_names())
        for name in ("sessions", "messages", "steps", "self_assessments", "mastery"):
            self.assertIn(name, tables)


class TestPersist(StoreCase):
    def _new_session(self, pid="limit"):
        sess, payload = Engine.create(problem_id=pid)
        store.save(sess)
        return sess

    def test_session_survives_cache_clear(self):
        sess = self._new_session()
        Engine.answer(sess, "0")
        store.save(sess)

        store.clear()                       # 模拟服务重启
        self.assertFalse(store.exists("不存在"))

        again = store.get(sess.id)
        self.assertEqual(again.id, sess.id)
        self.assertEqual(again.node_idx, 1)
        self.assertEqual(len(again.results), 1)
        self.assertEqual(again.problem_id, "limit")

    def test_history_is_restored(self):
        sess = self._new_session()
        Engine.answer(sess, "0")
        store.save(sess)
        n = len(sess.history)

        store.clear()
        again = store.get(sess.id)
        self.assertEqual(len(again.history), n)
        self.assertEqual(again.history[0]["role"], "assistant")
        roles = [m["role"] for m in again.history]
        self.assertIn("student", roles)
        self.assertEqual(again.history[-1]["text"], sess.history[-1]["text"])

    def test_can_continue_after_restart(self):
        sess = self._new_session()
        Engine.answer(sess, "0")
        store.save(sess)

        store.clear()
        again = store.get(sess.id)
        res = Engine.answer(again, "0/0")
        store.save(again)
        self.assertEqual(res["verdict"], "correct")
        self.assertEqual(again.node_idx, 2)

        store.clear()
        third = store.get(sess.id)
        self.assertEqual(third.node_idx, 2, "重启后进度不能退回去")

    def test_steps_table_matches_nodes(self):
        from app.models import StepRow
        sess = self._new_session("eigen")
        with __import__("app.db", fromlist=["SessionLocal"]).SessionLocal() as s:
            rows = list(s.scalars(select(StepRow).where(StepRow.session_id == sess.id)))
        self.assertEqual(len(rows), len(sess.nodes))
        self.assertEqual(sorted(r.n for r in rows), [1, 2, 3])
        self.assertTrue(all(r.question for r in rows))

    def test_mastery_written_only_for_answered(self):
        from app.models import MasteryRow
        from app.db import SessionLocal
        sess = self._new_session()
        Engine.answer(sess, "0")
        store.save(sess)
        with SessionLocal() as s:
            rows = list(s.scalars(select(MasteryRow).where(MasteryRow.session_id == sess.id)))
        self.assertEqual(len(rows), 1, "只答了一个节点，只该有一行掌握度")
        self.assertEqual(rows[0].knowledge, "直接代入")
        self.assertEqual(rows[0].score, 100)
        self.assertEqual((rows[0].correct, rows[0].total), (1, 1))

    def test_self_assessment_row_after_variant(self):
        from app.models import SelfAssessmentRow
        from app.db import SessionLocal
        sess = self._new_session()
        walk(sess)
        Engine.self_assess(sess, 5)
        store.save(sess)
        Engine.make_variant(sess)
        Engine.judge_variant(sess, "1/6")
        store.save(sess)

        with SessionLocal() as s:
            rows = list(s.scalars(select(SelfAssessmentRow).where(SelfAssessmentRow.session_id == sess.id)))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].score, 5)
        self.assertEqual(rows[0].actual, 5)
        self.assertEqual(rows[0].gap, 0)
        self.assertFalse(rows[0].warned, "全对却自评满分，不该亮红灯")

    def test_status_transitions(self):
        from app.models import SessionRow
        from app.db import SessionLocal
        sess = self._new_session()
        with SessionLocal() as s:
            self.assertEqual(s.get(SessionRow, sess.id).status, "running")
        walk(sess)
        store.save(sess)
        with SessionLocal() as s:
            self.assertEqual(s.get(SessionRow, sess.id).status, "finished")
        Engine.self_assess(sess, 3)
        store.save(sess)
        with SessionLocal() as s:
            self.assertEqual(s.get(SessionRow, sess.id).status, "assessed")

    def test_count_and_all(self):
        before = store.count()
        self._new_session("integral")
        self.assertEqual(store.count(), before + 1)
        self.assertTrue(any(s.problem_id == "integral" for s in store.all()))


class TestMemoryStore(unittest.TestCase):
    def test_same_interface(self):
        from app.core.store import MemoryStore
        m = MemoryStore()
        sess, _ = Engine.create(problem_id="limit")
        m.save(sess)
        self.assertEqual(m.get(sess.id).id, sess.id)
        self.assertEqual(m.count(), 1)
        m.clear()
        self.assertEqual(m.count(), 0)
        with self.assertRaises(Exception):
            m.get(sess.id)


if __name__ == "__main__":
    unittest.main()
