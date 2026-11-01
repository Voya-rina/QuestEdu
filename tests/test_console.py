# -*- coding: utf-8 -*-
"""控制台（Dashboard）测试：页面按角色分流 + API 权限矩阵 + 心跳。

TestClient 的对端是 testserver（控制端白名单），带 X-Forwarded-For 头
即被判定为用户端——用它模拟扫码进来的学生设备。
"""
from __future__ import annotations

import unittest
import unittest.mock
from unittest import mock

from fastapi.testclient import TestClient

from app.core import access, console_data
from app.main import app

client = TestClient(app)


def student_headers():
    """模拟已输过访问码的学生设备：转发头（服务端判定为用户端）+ 闸门 cookie。"""
    return {"X-Forwarded-For": "203.0.113.9",
            "Cookie": f"{access.COOKIE}={access.cookie_value()}"}


class TestRoleRouting(unittest.TestCase):
    def test_root_for_controller_is_console(self):
        r = client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("控制台", r.text)

    def test_root_for_student_is_student_page(self):
        r = client.get("/", headers=student_headers())
        self.assertEqual(r.status_code, 200)
        self.assertIn("QuestEdu", r.text)
        self.assertNotIn("CONSOLE 控制台", r.text)

    def test_console_page_redirects_students(self):
        r = client.get("/console", headers=student_headers(), follow_redirects=False)
        self.assertEqual(r.status_code, 302)
        self.assertEqual(r.headers["location"], "/")

    def test_student_page_open_for_everyone(self):
        self.assertEqual(client.get("/student").status_code, 200)
        self.assertEqual(client.get("/student", headers=student_headers()).status_code, 200)


class TestConsoleAPI(unittest.TestCase):
    def test_overview_rejects_students(self):
        r = client.get("/api/console/overview", headers=student_headers())
        self.assertEqual(r.status_code, 403)

    def test_overview_for_controller(self):
        r = client.get("/api/console/overview")
        self.assertEqual(r.status_code, 200)
        d = r.json()
        for key in ("llm", "gate", "devices", "sessions", "known_devices",
                    "access", "stats"):
            self.assertIn(key, d)
        self.assertIn("code", d["gate"])
        # QR 展示卡的数据源：三种入口地址 + 教学统计的字段完整性
        for key in ("lan_candidates", "ipv6_url", "tunnel_url"):
            self.assertIn(key, d["access"])
        for key in ("today", "running", "finish_rate", "calib", "hotspots"):
            self.assertIn(key, d["stats"])

    def test_llm_log_and_logs_permission(self):
        self.assertEqual(client.get("/api/console/llm-log").status_code, 200)
        self.assertEqual(client.get("/api/console/logs").status_code, 200)
        self.assertEqual(client.get("/api/console/llm-log", headers=student_headers()).status_code, 403)
        self.assertEqual(client.get("/api/console/logs", headers=student_headers()).status_code, 403)

    def test_sessions_endpoint_permission(self):
        self.assertEqual(client.get("/api/console/sessions").status_code, 200)
        self.assertEqual(client.get("/api/console/sessions", headers=student_headers()).status_code, 403)


class TestSessionMonitor(unittest.TestCase):
    """会话实时监控：点开最近会话看对话流。"""

    @classmethod
    def setUpClass(cls):
        # 钉 mock 后直接走引擎建会话：不走 HTTP 建会话那条路（local 模式下
        # 会真调本地模型拆题，一次 27 秒，测试套件等不起）
        from tests.support import expected_answer, force_mock, release_mock
        cls._release = release_mock
        force_mock()
        from app.core.engine import Engine
        from app.core.store import store
        sess, _ = Engine.create(problem_id="limit", user="监控员")
        Engine.answer(sess, expected_answer(sess))       # 学生答一步，留下对话
        store.save(sess)
        cls.sid = sess.id

    @classmethod
    def tearDownClass(cls):
        cls._release()

    def test_detail_for_controller(self):
        r = client.get(f"/api/console/session/{self.sid}")
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertEqual(d["id"], self.sid)
        self.assertEqual(d["user"], "监控员")
        self.assertIn("messages", d)
        roles = {m["role"] for m in d["messages"]}
        self.assertIn("assistant", roles)          # 首问在场
        self.assertIn("student", roles)            # 刚提交的那句在场

    def test_detail_404_for_unknown(self):
        self.assertEqual(client.get("/api/console/session/nope").status_code, 404)

    def test_detail_rejects_students(self):
        self.assertEqual(
            client.get(f"/api/console/session/{self.sid}",
                       headers=student_headers()).status_code, 403)


class TestModeSwitch(unittest.TestCase):
    """运行方案快捷切换：只动 INFERENCE_MODE，其余配置原样保留。"""

    def setUp(self):
        # save_config 会写 .env——测试里把它指到临时文件，绝不碰真配置
        import tempfile
        from pathlib import Path
        self._tmp = tempfile.NamedTemporaryFile(
            suffix=".env", delete=False, mode="w", encoding="utf-8")
        self._tmp.close()
        self._patcher = unittest.mock.patch(
            "app.core.llm.ENV_PATH", Path(self._tmp.name))
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        import os
        os.unlink(self._tmp.name)

    def test_switch_roundtrip(self):
        before = client.get("/api/llm").json()
        r = client.post("/api/console/mode", json={"mode": "local"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["mode"], "local")
        self.assertEqual(before["local_model"], r.json()["local_model"])
        back = client.post("/api/console/mode", json={"mode": "mock"})
        self.assertEqual(back.json()["mode"], "mock")

    def test_invalid_mode_rejected(self):
        self.assertEqual(
            client.post("/api/console/mode", json={"mode": "turbo"}).status_code, 422)

    def test_students_rejected(self):
        self.assertEqual(
            client.post("/api/console/mode", json={"mode": "local"},
                        headers=student_headers()).status_code, 403)


class TestDevicePerm(unittest.TestCase):
    """设备权限：默认 mock，可切 server；写盘文件在测试里指向临时路径。"""

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls._tmp = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w", encoding="utf-8")
        cls._tmp.close()
        cls._patcher = unittest.mock.patch.object(
            console_data, "_PERM_FILE", __import__("pathlib").Path(cls._tmp.name))
        cls._patcher.start()
        console_data._PERMS.clear()

    @classmethod
    def tearDownClass(cls):
        cls._patcher.stop()
        import os
        os.unlink(cls._tmp.name)
        console_data._PERMS.clear()

    def test_default_is_mock_and_roundtrip(self):
        self.assertEqual(console_data.device_perm("10.9.9.9"), "mock")
        r = client.post("/api/console/device-perm",
                        json={"ip": "10.9.9.9", "perm": "server"})
        self.assertEqual(r.json()["perm"], "server")
        self.assertEqual(console_data.device_perm("10.9.9.9"), "server")
        r = client.post("/api/console/device-perm",
                        json={"ip": "10.9.9.9", "perm": "mock"})
        self.assertEqual(r.json()["perm"], "mock")

    def test_invalid_perm_rejected(self):
        self.assertEqual(
            client.post("/api/console/device-perm",
                        json={"ip": "10.9.9.9", "perm": "root"}).status_code, 422)

    def test_students_rejected(self):
        self.assertEqual(
            client.post("/api/console/device-perm",
                        json={"ip": "10.9.9.9", "perm": "server"},
                        headers=student_headers()).status_code, 403)

    def test_devices_endpoint_lists_perm(self):
        # 授权跟随 IP 持久化：掉线重连（超过在线窗口）权限保持不变
        console_data.touch("10.9.9.9")                          # 先上线
        console_data.set_device_perm("10.9.9.9", "server")      # 在线期间授权
        console_data.touch("10.9.9.9")                          # 持续在线：保留
        d = client.get("/api/console/devices").json()
        row = next(x for x in d["devices"] if x["ip"] == "10.9.9.9")
        self.assertEqual(row["perm"], "server")
        # 模拟掉线重连：把最后心跳拨回在线窗口之外
        console_data._devices["10.9.9.9"]["seen"] -= console_data.ONLINE_WINDOW + 1
        console_data.touch("10.9.9.9")
        d = client.get("/api/console/devices").json()
        row = next(x for x in d["devices"] if x["ip"] == "10.9.9.9")
        self.assertEqual(row["perm"], "server")
        # 配置权限端点能看到这台设备；收回后消失
        perms = client.get("/api/console/perms").json()["devices"]
        self.assertIn("10.9.9.9", [x["ip"] for x in perms])
        client.post("/api/console/device-perm",
                    json={"ip": "10.9.9.9", "perm": "mock"})
        perms = client.get("/api/console/perms").json()["devices"]
        self.assertNotIn("10.9.9.9", [x["ip"] for x in perms])


class TestSessionDelete(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.support import force_mock, release_mock
        cls._release = release_mock
        force_mock()
        from app.core.engine import Engine
        from app.core.store import store
        sess, _ = Engine.create(problem_id="limit", user="待删")
        store.save(sess)
        cls.sid = sess.id

    @classmethod
    def tearDownClass(cls):
        cls._release()

    def test_delete_then_404(self):
        self.assertEqual(client.delete(f"/api/console/session/{self.sid}").status_code, 200)
        self.assertEqual(client.get(f"/api/console/session/{self.sid}").status_code, 404)

    def test_deleted_session_not_resurrected_by_late_save(self):
        """删除后学生端在途请求晚到落盘，不得把会话写回库。"""
        from fastapi import HTTPException

        from app.core.engine import Engine
        from app.core.store import store
        sess, _ = Engine.create(problem_id="limit", user="在途")
        store.save(sess)
        store.delete(sess.id)
        store.save(sess)          # 模拟在途 Engine.answer 完成后的整体落盘
        with self.assertRaises(HTTPException):
            store.get(sess.id)

    def test_students_rejected(self):
        self.assertEqual(
            client.delete("/api/console/session/nope",
                          headers=student_headers()).status_code, 403)


class TestSessionPurge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.support import force_mock, release_mock, walk
        cls._release = release_mock
        force_mock()
        from app.core.engine import Engine
        from app.core.store import store
        cls.store = store
        s1, _ = Engine.create(problem_id="limit", user="清理甲")
        store.save(s1)
        s2, _ = Engine.create(problem_id="limit", user="清理乙")
        walk(s2)
        store.save(s2)
        cls.running_id, cls.finished_id = s1.id, s2.id

    @classmethod
    def tearDownClass(cls):
        cls._release()
        cls.store.delete(cls.running_id)
        cls.store.delete(cls.finished_id)

    def _exists(self, sid):
        try:
            self.store.get(sid)
            return True
        except Exception:
            return False

    def test_purge_scopes(self):
        # 三个 scope 有先后依赖，放同一个用例按序执行
        r = client.post("/api/console/sessions/purge", json={"scope": "finished"})
        self.assertGreaterEqual(r.json()["deleted"], 1)
        self.assertFalse(self._exists(self.finished_id))
        self.assertTrue(self._exists(self.running_id))

        self.assertTrue(self._exists(self.running_id))
        r = client.post("/api/console/sessions/purge", json={"scope": "all"})
        self.assertGreaterEqual(r.json()["deleted"], 1)
        self.assertFalse(self._exists(self.running_id))

    def test_purge_older_requires_hours(self):
        self.assertEqual(
            client.post("/api/console/sessions/purge",
                        json={"scope": "older"}).status_code, 400)


class TestDevicePermAffectsSessionMode(unittest.TestCase):
    """mock 权限设备的建会话请求必须以 mode=mock 走引擎。"""

    def test_perm_controls_create_mode(self):
        from tests.support import force_mock, release_mock
        force_mock()
        try:
            from app.core.engine import Engine
            from app.api import routes_session
            sess, payload = Engine.create(problem_id="limit")
            with mock.patch.object(routes_session.console_data, "device_perm",
                                   return_value="mock"), \
                 mock.patch.object(routes_session.Engine, "create",
                                   return_value=(sess, payload)) as m:
                client.post("/api/sessions",
                            json={"problem_id": "limit"}, headers=student_headers())
                self.assertEqual(m.call_args.kwargs.get("mode"), "mock")
            with mock.patch.object(routes_session.console_data, "device_perm",
                                   return_value="server"), \
                 mock.patch.object(routes_session.Engine, "create",
                                   return_value=(sess, payload)) as m:
                client.post("/api/sessions",
                            json={"problem_id": "limit"}, headers=student_headers())
                self.assertEqual(m.call_args.kwargs.get("mode"), "")
        finally:
            release_mock()


class TestHeartbeat(unittest.TestCase):
    def test_heartbeat_records_device(self):
        r = client.post("/api/heartbeat",
                        json={"sid": "", "cur": 2, "total": 6})
        self.assertEqual(r.status_code, 200)
        d = client.get("/api/console/overview").json()
        rows = [x for x in d["devices"] if x.get("label", "").startswith("#")]
        self.assertTrue(rows, "心跳后设备应出现在在线列表且带 # 标签")
        self.assertEqual(rows[0]["cur"], 2)
        self.assertTrue(rows[0]["online"])

    def test_heartbeat_accepts_minimal_body(self):
        r = client.post("/api/heartbeat", json={})
        self.assertEqual(r.status_code, 200)


if __name__ == "__main__":
    unittest.main()
