# -*- coding: utf-8 -*-
"""访问入口测试：访问码闸门 + 隧道状态机。

这里不真的去连 Cloudflare——测的是"谁该被拦、谁能过"和"没有 cloudflared 时怎么报"。
真起隧道是网络行为，放进手工验收。
"""
from __future__ import annotations

import os
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from app.core import access


class AccessCase(unittest.TestCase):
    def setUp(self):
        if access.CONFIG_PATH.exists():
            access.CONFIG_PATH.unlink()
        access._attempts.clear()

    def tearDown(self):
        if access.CONFIG_PATH.exists():
            access.CONFIG_PATH.unlink()
        access._attempts.clear()


class TestCode(AccessCase):
    def test_code_is_four_digits(self):
        c = access.code()
        self.assertEqual(len(c), 4)
        self.assertTrue(c.isdigit())

    def test_code_is_stable(self):
        self.assertEqual(access.code(), access.code())

    def test_set_code_validates(self):
        with self.assertRaises(access.AccessError):
            access.set_code("12a4")
        with self.assertRaises(access.AccessError):
            access.set_code("12345")
        self.assertEqual(access.set_code("0007"), "0007")
        self.assertEqual(access.code(), "0007")

    def test_cookie_follows_code(self):
        old = access.cookie_value()
        self.assertTrue(access.cookie_ok(old))
        self.assertFalse(access.cookie_ok("nope"))
        access.set_code("9999")
        self.assertFalse(access.cookie_ok(old), "换了访问码，旧 cookie 必须失效")

    def test_host_url_validation(self):
        self.assertEqual(access.set_host_url("https://x.app.workbuddy.host/?sharecode=1"),
                         "https://x.app.workbuddy.host/?sharecode=1")
        self.assertEqual(access.host_url(), "https://x.app.workbuddy.host/?sharecode=1")
        with self.assertRaises(access.AccessError):
            access.set_host_url("x.app.workbuddy.host")


class TestLocalHost(AccessCase):
    def test_local_hosts(self):
        for h in ("127.0.0.1:8062", "localhost:8062", "[::1]:8062", "testserver"):
            self.assertTrue(access.is_local_host(h), h)

    def test_public_hosts(self):
        for h in ("abc-def.trycloudflare.com", "x.app.workbuddy.host", "example.com:443"):
            self.assertFalse(access.is_local_host(h), h)

    def test_no_host_header_is_local(self):
        self.assertTrue(access.is_local_host(""))

    def test_forwarded_request_always_needs_code(self):
        """伪造 Host 绕不过去：有转发头就一定要输码。

        实打实的绕过路径：`curl -H "Host: 127.0.0.1" https://xxx.trycloudflare.com`
        ——只看 Host 的话，这就等于本机直连。
        """
        headers = {"x-forwarded-for": "1.2.3.4"}
        self.assertTrue(access.needs_code("127.0.0.1:8062", headers))
        self.assertTrue(access.needs_code("abc.trycloudflare.com", {}))
        self.assertFalse(access.needs_code("127.0.0.1:8062", {}))


class TestTunnel(AccessCase):
    def test_no_binary_reports_how_to_fix(self):
        with mock.patch.object(access, "find_cloudflared", return_value=""):
            t = access.Tunnel()
            snap = t.start(8062)
        self.assertEqual(snap["state"], "error")
        self.assertIn("cloudflared", snap["message"])

    def test_binary_from_env(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "cloudflared.exe"
            exe.write_text("", encoding="utf-8")
            with mock.patch.dict(os.environ, {"CLOUDFLARED": str(exe)}):
                self.assertEqual(access.find_cloudflared(), str(exe))

    def test_off_by_default(self):
        snap = access.Tunnel().snapshot()
        self.assertEqual(snap["state"], "off")
        self.assertEqual(snap["url"], "")


class TestGate(AccessCase):
    """闸门挂在 ASGI 层，用真实的 app 走一遍。"""

    def setUp(self):
        super().setUp()
        from app.main import app
        self.client = TestClient(app)
        self.code = access.code()

    def _h(self, host):
        return {"host": host}

    def test_local_host_passes(self):
        r = self.client.get("/api/health", headers=self._h("127.0.0.1:8062"))
        self.assertEqual(r.status_code, 200)

    def test_public_host_gets_gate_page(self):
        r = self.client.get("/", headers=self._h("abc.trycloudflare.com"), follow_redirects=False)
        self.assertEqual(r.status_code, 401)
        self.assertIn("访问码", r.text)

    def test_public_api_is_blocked(self):
        r = self.client.get("/api/health", headers=self._h("abc.trycloudflare.com"))
        self.assertEqual(r.status_code, 401)
        self.assertIn("访问码", r.json()["detail"])

    def test_wrong_code_rejected(self):
        r = self.client.post("/__gate", data={"code": "0000" if self.code != "0000" else "1111"},
                             headers=self._h("abc.trycloudflare.com"), follow_redirects=False)
        self.assertEqual(r.status_code, 401)

    def test_right_code_sets_cookie_and_opens(self):
        r = self.client.post("/__gate", data={"code": self.code},
                             headers=self._h("abc.trycloudflare.com"), follow_redirects=False)
        self.assertEqual(r.status_code, 303)
        self.assertIn(access.COOKIE, r.cookies)
        # 带上 cookie 再看，应该放行
        r2 = self.client.get("/api/health", headers=self._h("abc.trycloudflare.com"))
        self.assertEqual(r2.status_code, 200)

    def test_static_assets_stay_open(self):
        """闸门页面本身要能加载（虽然它是内联样式），静态资源不拦。"""
        r = self.client.get("/static/manifest.json", headers=self._h("abc.trycloudflare.com"))
        self.assertNotEqual(r.status_code, 401)

    def test_rate_limit(self):
        for _ in range(access._MAX_TRIES):
            access.note_try("1.2.3.4")
        self.assertTrue(access.too_many_tries("1.2.3.4"))
        self.assertFalse(access.too_many_tries("5.6.7.8"))


class TestIPv6(AccessCase):
    class FakeSock:
        def __init__(self, addr):
            self.addr = addr

        def connect(self, *_):
            pass

        def getsockname(self):
            return (self.addr, 0, 0, 0)

        def close(self):
            pass

    def _with(self, addr):
        return mock.patch.object(access.socket, "socket", lambda *a, **k: self.FakeSock(addr))

    def test_global_address_is_used(self):
        with self._with("2409:8746:40:1b95::1"):
            self.assertEqual(access.ipv6_global(), "2409:8746:40:1b95::1")

    def test_link_local_and_ula_are_rejected(self):
        for bad in ("fe80::1", "fc00::1", "fd12::3", "::1", "::"):
            with self._with(bad):
                self.assertEqual(access.ipv6_global(), "", bad)

    def test_describe_has_ipv6_block(self):
        with self._with("2409:8746:40:1b95::1"):
            d = access.describe(8062)
        self.assertEqual(d["ipv6"]["url"], "http://[2409:8746:40:1b95::1]:8062")

    def test_qr_target_only_for_own_hosts(self):
        self.assertTrue(access.safe_qr_target("http://127.0.0.1:8062/"))
        self.assertTrue(access.safe_qr_target("https://abc.trycloudflare.com/"))
        self.assertTrue(access.safe_qr_target("https://x.app.workbuddy.host/?sharecode=1"))
        self.assertFalse(access.safe_qr_target("https://example.com/"))
        self.assertFalse(access.safe_qr_target("javascript:alert(1)"))
        self.assertFalse(access.safe_qr_target("http://"))


class TestDescribe(AccessCase):
    def test_shape(self):
        d = access.describe(8062)
        self.assertIn("lan", d)
        self.assertIn("tunnel", d)
        self.assertIn("host", d)
        self.assertTrue(d["lan"]["url"].startswith("http://"))
        self.assertEqual(len(d["gate"]["code"]), 4)


if __name__ == "__main__":
    unittest.main()
