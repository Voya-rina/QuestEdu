# -*- coding: utf-8 -*-
"""拍照 OCR 测试。

重点不在"能不能认字"（那要真模型才知道），而在**识别失败时怎么办**：
不能把模型瞎猜的式子当成学生的题目送进追问流程。
"""
from __future__ import annotations

import unittest
from unittest import mock

from app.core import ocr

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 200
JPG = b"\xff\xd8\xff" + b"\x00" * 200


class TestPreflight(unittest.TestCase):
    """还没轮到模型说话就该挡掉的输入。"""

    def test_empty(self):
        r = ocr.extract(b"", "image/png")
        self.assertFalse(r["ok"])
        self.assertEqual(r["reason"], "empty")

    def test_too_large(self):
        r = ocr.extract(b"\x89PNG\r\n\x1a\n" + b"0" * (ocr.MAX_BYTES + 10), "image/png")
        self.assertFalse(r["ok"])
        self.assertEqual(r["reason"], "too_large")

    def test_mime_sniffing(self):
        self.assertEqual(ocr._sniff_mime(PNG, "application/octet-stream"), "image/png")
        self.assertEqual(ocr._sniff_mime(JPG, None), "image/jpeg")
        self.assertEqual(ocr._sniff_mime(b"RIFF0000WEBPVP8", "x"), "image/webp")

    def test_bad_type(self):
        r = ocr.extract(b"hello world", "text/plain")
        self.assertFalse(r["ok"])
        self.assertEqual(r["reason"], "bad_type")

    def test_no_model_configured(self):
        """默认 mock 模式下没有多模态模型：要给学生一句能照做的话。"""
        r = ocr.extract(PNG, "image/png")
        self.assertFalse(r["ok"])
        self.assertEqual(r["reason"], "no_model")
        self.assertIn("打进来", r["message"])


class TestModelPath(unittest.TestCase):
    """模型这一层用打桩替换，验的是我们自己写的编排。"""

    def _run(self, reply, raises=None):
        with mock.patch.object(ocr.llm, "vision_available", return_value=True), \
             mock.patch.object(ocr.llm, "chat_vision", side_effect=raises, return_value=reply):
            return ocr.extract(PNG, "image/png")

    def test_success(self):
        r = self._run('{"found":true,"latex":"$\\\\int_0^1 x^2 dx$","note":"求定积分"}')
        self.assertTrue(r["ok"])
        self.assertIn("int_0^1", r["latex"])
        self.assertEqual(r["note"], "求定积分")

    def test_fenced_json_is_accepted(self):
        r = self._run('```json\n{"found":true,"latex":"$x^2$"}\n```')
        self.assertTrue(r["ok"])
        self.assertEqual(r["latex"], "$x^2$")

    def test_not_found(self):
        r = self._run('{"found":false,"latex":"","note":"是一段风景照"}')
        self.assertFalse(r["ok"])
        self.assertEqual(r["reason"], "not_found")
        self.assertIn("没认出", r["message"])

    def test_empty_latex_counts_as_not_found(self):
        r = self._run('{"found":true,"latex":"   "}')
        self.assertFalse(r["ok"])
        self.assertEqual(r["reason"], "not_found")

    def test_garbage_reply_is_not_trusted(self):
        """模型吐了一堆没法解析的东西，宁可说失败，也不能把原文当题目。"""
        r = self._run("抱歉，我看不清这张图。")
        self.assertFalse(r["ok"])
        self.assertEqual(r["reason"], "not_found")

    def test_api_error_is_reported(self):
        r = self._run(None, raises=RuntimeError("HTTP 401 · invalid key"))
        self.assertFalse(r["ok"])
        self.assertEqual(r["reason"], "api_error")
        self.assertIn("打进来", r["message"])

    def test_timeout_is_marked_retryable(self):
        r = self._run(None, raises=ocr.llm.LLMTimeout("等了 900 秒还没回话（本地模型首次加载权重可能要一两分钟）"))
        self.assertFalse(r["ok"])
        self.assertEqual(r["reason"], "timeout")
        self.assertTrue(r["retryable"], "超时该提示再等一下，而不是说这条路走不通")
        self.assertIn("稍等", r["message"])

    def test_image_unsupported_is_explained(self):
        """模型端的 400 原文甩给用户等于什么都没说。"""
        r = self._run(None, raises=RuntimeError(
            "HTTP 400 · this model does not support image input"))
        self.assertFalse(r["ok"])
        self.assertIn("不支持看图", r["message"])


class TestLocalVision(unittest.TestCase):
    """本地文本模型不能看图——用户实测 qwen3.8 就是这个情况。"""

    def test_local_text_model_blocks_ocr_with_reason(self):
        from app.core import llm
        saved = dict(llm._cfg)
        try:
            llm._cfg.update(mode="local", local_base_url="http://127.0.0.1:1234/v1",
                            local_model="qwen3.8", local_vision_model="",
                            vision_model="")
            r = ocr.extract(PNG, "image/png")
            self.assertFalse(r["ok"])
            self.assertEqual(r["reason"], "no_model")
            self.assertIn("多模态", r["why"], "要说明缺的是多模态模型，而不是笼统说没接模型")
        finally:
            llm._cfg.clear()
            llm._cfg.update(saved)


if __name__ == "__main__":
    unittest.main()
