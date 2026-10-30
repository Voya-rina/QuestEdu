# -*- coding: utf-8 -*-
"""统一 LLM 适配层测试：三种模式的端点选择与回落。

这里不真的发请求——测的是"该连谁、连不上怎么办"，那才是我们自己写的逻辑。
"""
from __future__ import annotations

import unittest
from unittest import mock

from app.core import llm


class LLMCase(unittest.TestCase):
    def setUp(self):
        self._saved = dict(llm._cfg)

    def tearDown(self):
        llm._cfg.clear()
        llm._cfg.update(self._saved)


class TestRoutes(LLMCase):
    def test_mock_has_no_route(self):
        llm._cfg.update(mode="mock")
        self.assertEqual(llm._routes(), [])
        self.assertFalse(llm.online())

    def test_cloud(self):
        llm._cfg.update(mode="cloud", base_url="https://api.example.com/v1", api_key="sk-x")
        self.assertEqual([r["name"] for r in llm._routes()], ["cloud"])
        self.assertTrue(llm.online())

    def test_cloud_without_key_is_not_ready(self):
        llm._cfg.update(mode="cloud", base_url="https://api.example.com/v1", api_key="")
        self.assertFalse(llm.online())

    def test_local_needs_no_real_key(self):
        """本地 llama-server 通常不校验 Key，但不能让请求头空着。"""
        llm._cfg.update(mode="local", local_base_url="http://127.0.0.1:8080/v1",
                        local_api_key="")
        routes = llm._routes()
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0]["name"], "local")
        self.assertEqual(routes[0]["api_key"], "sk-local")
        self.assertTrue(llm.online())

    def test_hybrid_prefers_cloud_then_local(self):
        llm._cfg.update(mode="hybrid", base_url="https://api.example.com/v1", api_key="sk-x",
                        local_base_url="http://127.0.0.1:8080/v1")
        self.assertEqual([r["name"] for r in llm._routes()], ["cloud", "local"])

    def test_hybrid_without_cloud_key_uses_local_only(self):
        llm._cfg.update(mode="hybrid", base_url="https://api.example.com/v1", api_key="",
                        local_base_url="http://127.0.0.1:8080/v1")
        self.assertEqual([r["name"] for r in llm._routes()], ["local"])


class TestFallback(LLMCase):
    def setUp(self):
        super().setUp()
        llm._cfg.update(mode="hybrid", base_url="https://api.example.com/v1", api_key="sk-x",
                        local_base_url="http://127.0.0.1:8080/v1")

    def test_falls_back_to_local(self):
        calls = []

        def fake_send(route, messages, temperature, max_tokens=None):
            calls.append(route["name"])
            if route["name"] == "cloud":
                raise llm.LLMError("HTTP 401 · invalid key")
            return "本地回答"

        with mock.patch.object(llm, "_send", side_effect=fake_send):
            out = llm._post([{"role": "user", "content": "hi"}], 0.2)

        self.assertEqual(out, "本地回答")
        self.assertEqual(calls, ["cloud", "local"], "应当先云端后本地")

    def test_reports_both_failures(self):
        with mock.patch.object(llm, "_send", side_effect=llm.LLMError("连不上")):
            with self.assertRaises(llm.LLMError) as ctx:
                llm._post([{"role": "user", "content": "hi"}], 0.2)
        msg = str(ctx.exception)
        self.assertIn("cloud", msg)
        self.assertIn("local", msg)

    def test_single_route_does_not_retry(self):
        llm._cfg.update(mode="cloud")
        calls = []

        def fake_send(route, messages, temperature, max_tokens=None):
            calls.append(route["name"])
            raise llm.LLMError("boom")

        with mock.patch.object(llm, "_send", side_effect=fake_send):
            with self.assertRaises(llm.LLMError):
                llm._post([{"role": "user", "content": "hi"}], 0.2)
        self.assertEqual(calls, ["cloud"])


class TestSaveConfig(LLMCase):
    """保存配置会把 .env 改写，测试必须指向临时文件，绝不能动真配置。"""

    def _save(self, **kw):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d) / ".env"
            with mock.patch.object(llm, "ENV_PATH", tmp):
                llm.save_config(**kw)
                return tmp.read_text(encoding="utf-8")

    def test_local_mode_persisted(self):
        text = self._save(mode="local", local_base_url="http://127.0.0.1:9000/v1",
                          local_model="qwen2.5-7b")
        self.assertIn("INFERENCE_MODE=local", text)
        self.assertIn("LLM_LOCAL_BASE_URL=http://127.0.0.1:9000/v1", text)
        self.assertIn("LLM_LOCAL_MODEL=qwen2.5-7b", text)

    def test_hybrid_mode_persisted(self):
        text = self._save(mode="hybrid", base_url="https://api.x.com/v1",
                          api_key="sk-1", model="deepseek-chat")
        self.assertIn("INFERENCE_MODE=hybrid", text)
        self.assertIn("LLM_API_KEY=sk-1", text)

    def test_blank_mode_falls_back_to_auto(self):
        self.assertIn("INFERENCE_MODE=cloud",
                      self._save(base_url="https://api.x.com/v1", api_key="sk-1"))
        self.assertIn("INFERENCE_MODE=mock", self._save(base_url="", api_key=""))

    def test_local_config_kept_when_saving_cloud(self):
        """切换模式不该把另一条链路的手工配置抹掉。"""
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d) / ".env"
            tmp.write_text("LLM_LOCAL_BASE_URL=http://127.0.0.1:9999/v1\nLLM_LOCAL_MODEL=my-local\n",
                           encoding="utf-8")
            with mock.patch.object(llm, "ENV_PATH", tmp):
                llm.save_config("https://api.x.com/v1", "sk-1", "m", mode="cloud")
            text = tmp.read_text(encoding="utf-8")
        self.assertIn("LLM_LOCAL_BASE_URL=http://127.0.0.1:9999/v1", text)
        self.assertIn("LLM_LOCAL_MODEL=my-local", text)

    def test_blank_key_keeps_saved_one(self):
        """密码框留空＝沿用已保存的 Key。

        这条是从实机踩出来的：界面上 Key 那格为了安全每次都是空的，
        以前把它当"清空"写回去，结果改个模型名就把 Key 弄丢了，
        下次还得重新贴一遍。
        """
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d) / ".env"
            tmp.write_text("INFERENCE_MODE=cloud\nLLM_API_KEY=sk-old\n", encoding="utf-8")
            with mock.patch.object(llm, "ENV_PATH", tmp):
                llm.save_config("https://api.x.com/v1", None, "m2", mode="cloud")
            text = tmp.read_text(encoding="utf-8")
        self.assertIn("LLM_API_KEY=sk-old", text)
        self.assertIn("LLM_MODEL=m2", text)

    def test_clear_key_explicitly(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d) / ".env"
            tmp.write_text("INFERENCE_MODE=cloud\nLLM_API_KEY=sk-old\n", encoding="utf-8")
            with mock.patch.object(llm, "ENV_PATH", tmp):
                llm.save_config("https://api.x.com/v1", None, "m", mode="cloud", clear_key=True)
            text = tmp.read_text(encoding="utf-8")
        self.assertIn("LLM_API_KEY=\n", text)

    def test_mock_mode_is_a_selectable_plan(self):
        """离线不是"清除配置"，是并列的第四种运行方案。"""
        self.assertIn("INFERENCE_MODE=mock", self._save(mode="mock", local_model="qwen3.8"))


class TestDescribe(LLMCase):
    def test_never_leaks_key(self):
        llm._cfg.update(mode="cloud", api_key="sk-abcdef123456789")
        d = llm.describe()
        self.assertNotIn("sk-abcdef123456789", str(d))
        self.assertNotIn("api_key", d)
        self.assertTrue(d["key_hint"].endswith("…"))

    def test_vision_requires_model(self):
        # 离线：什么都不能看
        llm._cfg.update(mode="mock")
        self.assertFalse(llm.vision_available())

        # 云端没单独配看图模型：允许拿主模型试一次（有些网关的模型本身吃图）
        llm._cfg.update(mode="cloud", base_url="https://x/v1", api_key="k",
                        vision_model="", local_vision_model="")
        self.assertTrue(llm.vision_available())
        llm._cfg.update(vision_model="qwen-vl")
        self.assertEqual(llm.vision_routes()[0]["model"], "qwen-vl")

    def test_local_vision_needs_explicit_model(self):
        """本地文本模型不能看图——这是实测踩到的坑，必须显式配多模态模型。"""
        llm._cfg.update(mode="local", local_base_url="http://127.0.0.1:1234/v1",
                        local_model="qwen3.8", local_vision_model="", vision_model="")
        self.assertFalse(llm.vision_available(), "qwen3.8 是文本模型，不该被当成能看图")
        self.assertIn("多模态", llm._vision_hint())

        llm._cfg.update(local_vision_model="qwen2.5-vl-7b")
        self.assertTrue(llm.vision_available())
        route = llm.vision_routes()[0]
        self.assertEqual(route["name"], "local")
        self.assertEqual(route["model"], "qwen2.5-vl-7b")


class TestContent(LLMCase):
    """思考型模型把正文放在 reasoning_content 里——这是实测日志里的真实形状。"""

    def test_reads_content(self):
        self.assertEqual(llm.content_of({"message": {"content": "好"}}), "好")

    def test_falls_back_to_reasoning(self):
        choice = {"message": {"content": "", "reasoning_content": '{"found":true}'},
                  "finish_reason": "stop"}
        self.assertEqual(llm.content_of(choice), '{"found":true}')

    def test_empty_both_raises(self):
        with self.assertRaises(llm.LLMError):
            llm.content_of({"message": {"content": "  "}, "finish_reason": "length"})


class TestTimeout(LLMCase):
    """超时和普通错误必须分开：一个只能等，一个可以重试。"""

    def _local(self, **kw):
        llm._cfg.update(mode="local", local_base_url="http://127.0.0.1:1234/v1",
                        local_model="qwen3.8", **kw)

    def test_timeout_is_not_retried(self):
        self._local()
        calls = []

        def fake(route, messages, temperature, max_tokens=None):
            calls.append(route["name"])
            raise llm.LLMTimeout("等了 900 秒还没回话")

        with mock.patch.object(llm, "_send", fake):
            with self.assertRaises(llm.LLMTimeout):
                llm.chat_json("s", "u", retries=3)
        self.assertEqual(len(calls), 1, "超时重试只会让模型在队列里排更久")

    def test_hybrid_mixed_error_is_not_timeout(self):
        llm._cfg.update(mode="hybrid", base_url="https://x/v1", api_key="k",
                        local_base_url="http://127.0.0.1:1234/v1")

        def fake(route, messages, temperature, max_tokens=None):
            if route["name"] == "cloud":
                raise llm.LLMError("HTTP 401")
            raise llm.LLMTimeout("等了 60 秒")

        with mock.patch.object(llm, "_send", fake):
            with self.assertRaises(llm.LLMError) as ctx:
                llm.chat("s", "u")
        self.assertNotIsInstance(ctx.exception, llm.LLMTimeout)
        self.assertIn("cloud", str(ctx.exception))
        self.assertIn("local", str(ctx.exception))

    def test_local_route_waits_longer_than_cloud(self):
        llm._cfg.update(mode="hybrid", base_url="https://x/v1", api_key="k",
                        local_base_url="http://127.0.0.1:1234/v1")
        r = {x["name"]: x for x in llm._routes()}
        self.assertGreater(r["local"]["timeout"], r["cloud"]["timeout"])


class TestParseJson(LLMCase):
    """模型在 JSON 里写 LaTeX 时不会转义反斜杠——这是最常踩的一个坑。"""

    def test_single_backslash_latex(self):
        d = llm.parse_json('{"e":"$\\lim_{x\\to 0}\\frac{1}{2}$"}')
        self.assertEqual(d["e"], "$\\lim_{x\\to 0}\\frac{1}{2}$")

    def test_frac_not_turned_into_formfeed(self):
        """`\\f` 是合法 JSON 转义，直接 json.loads 会静默变成换页符。"""
        d = llm.parse_json('{"e":"$\\frac{1}{2}$"}')
        self.assertEqual(d["e"], "$\\frac{1}{2}$")
        self.assertNotIn("\x0c", d["e"])

    def test_properly_escaped_untouched(self):
        d = llm.parse_json('{"e":"$\\\\frac{1}{2}$"}')
        self.assertEqual(d["e"], "$\\frac{1}{2}$")

    def test_markdown_fence(self):
        d = llm.parse_json('```json\n{"e":"$\\tan x$"}\n```')
        self.assertEqual(d["e"], "$\\tan x$")

    def test_surrounded_by_prose(self):
        d = llm.parse_json('结果如下：\n{"a":1}\n以上。')
        self.assertEqual(d, {"a": 1})

    def test_real_newline_survives(self):
        d = llm.parse_json('{"e":"第一行\\n第二行"}')
        self.assertIn("\n", d["e"])

    def test_unicode_escape_survives(self):
        self.assertEqual(llm.parse_json('{"e":"\\u6781\\u9650"}')["e"], "极限")

    def test_garbage_returns_none(self):
        self.assertIsNone(llm.parse_json("完全不是 JSON"))


class TestBaseUrl(LLMCase):
    """LM Studio 的地址通常只到端口，/v1 要自己补。"""

    def test_appends_v1(self):
        self.assertEqual(llm.norm_base("http://127.0.0.1:1234"), "http://127.0.0.1:1234/v1")

    def test_keeps_existing_v1(self):
        self.assertEqual(llm.norm_base("http://127.0.0.1:1234/v1/"), "http://127.0.0.1:1234/v1")

    def test_strips_completion_path(self):
        self.assertEqual(
            llm.norm_base("https://api.deepseek.com/v1/chat/completions"),
            "https://api.deepseek.com/v1")

    def test_blank_stays_blank(self):
        self.assertEqual(llm.norm_base("  "), "")


class TestDefaults(LLMCase):
    def test_local_defaults_point_at_lm_studio(self):
        self.assertEqual(llm._cfg["local_base_url"], "http://127.0.0.1:1234/v1")
        self.assertEqual(llm._cfg["local_model"], "qwen3.8")

    def test_local_timeout_is_generous(self):
        """本地思考型模型一次要几分钟，60 秒会把已经生成完的回答掐断。"""
        self.assertGreaterEqual(llm._cfg["local_timeout"], 300)


if __name__ == "__main__":
    unittest.main()
