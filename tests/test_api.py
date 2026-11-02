# -*- coding: utf-8 -*-
"""HTTP 层测试：路由、校验、错误码，以及静态资源是否真的挂上了。

前端全部依赖 /static（含本地 KaTeX），这一层挂了页面就是白屏，
所以静态文件也要测。
"""
from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.data.problems import PROBLEMS
from app.main import app
from tests.support import force_mock, release_mock

client = TestClient(app)


class APICase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        force_mock()

    @classmethod
    def tearDownClass(cls):
        release_mock()


class TestMeta(APICase):
    def test_index_serves_html(self):
        r = client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/html", r.headers["content-type"])
        self.assertIn("学问", r.text)

    def test_health(self):
        r = client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["ok"])

    def test_problems_are_categorised(self):
        data = client.get("/api/problems").json()["problems"]
        cats = {p["category"] for p in data}
        self.assertEqual(cats, {"微积分", "线性代数"})
        by_id = {p["id"]: p for p in data}
        self.assertEqual(by_id["eigen"]["category"], "线性代数")
        self.assertEqual(by_id["limit"]["category"], "微积分")

    def test_info_lists_all_problems(self):
        r = client.get("/api/info")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(len(data["problems"]), len(PROBLEMS))
        self.assertIn("llm", data)
        self.assertNotIn("_answer", str(data))

    def test_openapi_available(self):
        self.assertEqual(client.get("/docs").status_code, 200)

    def test_pwa_manifest(self):
        r = client.get("/manifest.json")
        self.assertEqual(r.status_code, 200)
        self.assertIn("manifest", r.headers["content-type"])
        self.assertIn("学问", r.text)

    def test_service_worker_at_root(self):
        """sw.js 必须在根路径，否则控制不到页面。"""
        r = client.get("/sw.js")
        self.assertEqual(r.status_code, 200)
        self.assertIn("javascript", r.headers["content-type"])
        self.assertEqual(r.headers.get("service-worker-allowed"), "/")
        self.assertIn("questedu", r.text)

    def test_icons_served(self):
        for path in ("/static/icons/icon-192.png", "/static/icons/icon-512.png",
                     "/static/icons/apple-touch-icon.png"):
            with self.subTest(path=path):
                r = client.get(path)
                self.assertEqual(r.status_code, 200)
                self.assertGreater(len(r.content), 500)

    def test_qr_svg(self):
        r = client.get("/api/qr")
        self.assertEqual(r.status_code, 200)
        self.assertIn("svg", r.headers["content-type"])
        self.assertIn(b"<svg", r.content)

    def test_static_vendor_chartjs(self):
        r = client.get("/static/vendor/chartjs/chart.umd.js")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"Chart.js", r.content[:200])

    def test_static_vendor_katex(self):
        for path in (
            "/static/vendor/katex/katex.min.js",
            "/static/vendor/katex/katex.min.css",
            "/static/vendor/katex/contrib/auto-render.min.js",
        ):
            with self.subTest(path=path):
                r = client.get(path)
                self.assertEqual(r.status_code, 200)
                self.assertGreater(len(r.content), 1000)


class TestSessionFlow(APICase):
    def _create(self, pid="limit"):
        r = client.post("/api/sessions", json={"problem_id": pid})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()["session_id"]

    def test_full_flow(self):
        sid = self._create()
        for text in ("0", "0/0", "泰勒", "2", "5", "x^2/2", "1/2"):
            r = client.post(f"/api/sessions/{sid}/messages", json={"text": text})
            self.assertEqual(r.status_code, 200, r.text)
            if text == "1/2":
                self.assertEqual(r.json()["next_action"], "self_assessment")

        r = client.post(f"/api/sessions/{sid}/self-assessment", json={"score": 5})
        self.assertEqual(r.status_code, 200)
        self.assertIn(r.json()["level"], ("red", "neutral", "green"))

        r = client.post(f"/api/sessions/{sid}/variant", json={})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["verified"])

        r = client.post(f"/api/sessions/{sid}/variant/answer", json={"text": "1/6"})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["correct"])

        r = client.get(f"/api/sessions/{sid}/report")
        self.assertEqual(r.status_code, 200)
        rep = r.json()
        self.assertIn("总体掌握度", rep)
        self.assertTrue(rep["bars"])

    def test_radar_endpoint(self):
        sid = self._create()
        for text in ("0", "0/0", "泰勒", "2", "5", "x^2/2", "1/2"):
            client.post(f"/api/sessions/{sid}/messages", json={"text": text})
        r = client.get(f"/api/reports/{sid}/radar")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(len(data["labels"]), 6)
        self.assertEqual(len(data["values"]), 6)
        self.assertTrue(data["enough"])

    def test_radar_unknown_session_404(self):
        self.assertEqual(client.get("/api/reports/nope/radar").status_code, 404)

    def test_history_lists_sessions(self):
        """首页「学习记录」：轻量列表，含刚做完的会话。"""
        sid = self._create()
        client.post(f"/api/sessions/{sid}/messages", json={"text": "0"})
        r = client.get("/api/reports/history")
        self.assertEqual(r.status_code, 200)
        items = r.json()["sessions"]
        self.assertTrue(items)
        mine = next(x for x in items if x["id"] == sid)
        self.assertEqual(mine["status"], "running")
        self.assertEqual(mine["nodes_done"], 1)
        self.assertGreaterEqual(mine["nodes_total"], 3)
        self.assertIn("problem", mine)
        self.assertIn("created_at", mine)

    def test_resume_after_restart(self):
        """清缓存= 服务重启，会话必须还能接着答。"""
        from app.core.store import store

        sid = self._create()
        client.post(f"/api/sessions/{sid}/messages", json={"text": "0"})
        store.clear()

        r = client.get(f"/api/sessions/{sid}")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["progress"], [2, 6])

        r = client.post(f"/api/sessions/{sid}/messages", json={"text": "0/0"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["verdict"], "correct")

    def test_export_csv(self):
        sid = self._create()
        for text in ("0", "0/0"):
            client.post(f"/api/sessions/{sid}/messages", json={"text": text})
        client.post(f"/api/sessions/{sid}/self-assessment", json={"score": 4})

        r = client.get(f"/api/reports/{sid}/export.csv")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/csv", r.headers["content-type"])
        self.assertIn(f'questedu-{sid}.csv', r.headers["content-disposition"])
        text = r.content.decode("utf-8")
        self.assertTrue(text.startswith("\ufeff"), "缺 BOM，Excel 打开会乱码")
        self.assertIn("知识点掌握", text)
        self.assertIn("作答记录", text)
        self.assertIn("未定式判断", text)

    def test_get_session_state(self):
        sid = self._create()
        r = client.get(f"/api/sessions/{sid}")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["progress"], [1, 6])
        self.assertFalse(data["finished"])

    def test_unknown_session_404(self):
        for method, path in (
            ("get", "/api/sessions/nope/report"),
            ("post", "/api/sessions/nope/messages"),
        ):
            with self.subTest(path=path):
                if method == "get":
                    r = client.get(path)
                else:
                    r = client.post(path, json={"text": "hi"})
                self.assertEqual(r.status_code, 404)

    def test_empty_text_rejected(self):
        sid = self._create()
        r = client.post(f"/api/sessions/{sid}/messages", json={"text": ""})
        self.assertEqual(r.status_code, 422)

    def test_score_out_of_range_rejected(self):
        sid = self._create()
        self.assertEqual(
            client.post(f"/api/sessions/{sid}/self-assessment", json={"score": 101}).status_code, 422)
        self.assertEqual(
            client.post(f"/api/sessions/{sid}/self-assessment", json={"score": -1}).status_code, 422)

    def test_variant_answer_before_making_variant(self):
        sid = self._create()
        r = client.post(f"/api/sessions/{sid}/variant/answer", json={"text": "1/2"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("error", r.json())


class TestOCR(APICase):
    def test_upload_without_model(self):
        """mock 模式下没有多模态模型：接口要回 200 + 人话，不是 500。"""
        r = client.post("/api/ocr", files={"file": ("q.png", b"\x89PNG\r\n\x1a\n" + b"0" * 64, "image/png")})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertFalse(data["ok"])
        self.assertEqual(data["reason"], "no_model")

    def test_upload_requires_file(self):
        self.assertEqual(client.post("/api/ocr").status_code, 422)


class TestLLMRoutes(APICase):
    def test_status(self):
        r = client.get("/api/llm")
        self.assertEqual(r.status_code, 200)
        self.assertIn("mode", r.json())
        self.assertNotIn("api_key", r.json())

    def test_key_never_returned(self):
        r = client.get("/api/llm")
        body = r.text
        self.assertNotIn("sk-", body)


if __name__ == "__main__":
    unittest.main()
