# -*- coding: utf-8 -*-
"""流式通道测试。

关键是两条：分片拼起来必须与整段回复**逐字相同**，以及 done 的载荷
必须和 POST 版本完全一致——否则前端两条路径要写两套渲染逻辑。
"""
from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.main import app
from tests.support import force_mock, release_mock

client = TestClient(app)


class WSCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        force_mock()

    @classmethod
    def tearDownClass(cls):
        release_mock()

    def _session(self):
        r = client.post("/api/sessions", json={"problem_id": "limit"})
        return r.json()["session_id"]

    def _stream(self, sid, text):
        deltas, done = [], None
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"session_id": sid, "text": text})
            while True:
                msg = ws.receive_json()
                if msg["type"] == "delta":
                    deltas.append(msg["text"])
                elif msg["type"] == "done":
                    done = msg
                    break
                elif msg["type"] == "error":
                    return deltas, msg
        return deltas, done


class TestStream(WSCase):
    def test_chunks_rebuild_the_message(self):
        sid = self._session()
        deltas, done = self._stream(sid, "0")
        self.assertIsNotNone(done)
        self.assertEqual(done["verdict"], "correct")
        self.assertEqual("".join(deltas), done["message"], "分片拼回来必须一字不差")
        self.assertGreater(len(deltas), 1, "应当真的分成多片")

    def test_payload_matches_http_version(self):
        sid = self._session()
        _, done = self._stream(sid, "0")
        http = client.post(f"/api/sessions/{sid}/messages", json={"text": "泰勒"}).json()
        for key in ("message", "stage", "note", "verdict", "node_done",
                    "next_action", "progress"):
            self.assertIn(key, done)
            self.assertIn(key, http)

    def test_state_is_persisted_after_stream(self):
        sid = self._session()
        self._stream(sid, "0")
        state = client.get(f"/api/sessions/{sid}").json()
        self.assertEqual(state["progress"], [2, 6])

    def test_multiple_turns_on_one_socket(self):
        sid = self._session()
        with client.websocket_connect("/ws/chat") as ws:
            for text in ("0", "0/0"):
                ws.send_json({"session_id": sid, "text": text})
                while ws.receive_json()["type"] != "done":
                    pass
        self.assertEqual(client.get(f"/api/sessions/{sid}").json()["progress"], [3, 6])

    def test_unknown_session_reports_error(self):
        _, msg = self._stream("nope", "0/0")
        self.assertEqual(msg["type"], "error")
        self.assertIn("不存在", msg["message"])

    def test_empty_text_reports_error(self):
        sid = self._session()
        _, msg = self._stream(sid, "   ")
        self.assertEqual(msg["type"], "error")


if __name__ == "__main__":
    unittest.main()
