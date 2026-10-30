# -*- coding: utf-8 -*-
"""拍照 → LaTeX（大纲第三节第 1 条）。

云端走多模态模型，端侧以后可以换 OXR-1.0；接口形状一致，换引擎只改这一层。

铁律：**识别不出来就说识别不出来**。不要把模型瞎猜的式子当成学生的题目送进
追问流程——那会让学生在"一道自己没拍的题"上答题，比识别失败糟糕得多。
"""
from __future__ import annotations

import base64

from app.core import llm

MAX_BYTES = 8 * 1024 * 1024          # 8MB，手机直出照片通常 2~4MB
ALLOWED_MIME = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/heic", "image/heif"}

SYSTEM = """你是题目转写器。把图片里的数学题抄成 LaTeX，不做任何解答。

只输出 JSON：
{"found":true,"latex":"题目的 LaTeX，行内公式用 $...$","note":"题目里的文字要求(没有就空串)"}

规则：
1. 只转写，不解释、不化简、不改数字、不补条件。
2. 看不清的字符用 \\square 占位，不要猜。
3. 图里没有数学题（是风景、书页正文、手写草稿无法辨认）时输出 {"found":false,"latex":"","note":"原因"}。
4. 中文题干用中文，公式用 LaTeX。"""


def unavailable(reason: str = "no_model") -> dict:
    return {
        "ok": False,
        "reason": reason,
        # 给学生看的话，只说下一步能做什么。why 是"为什么不能"，直接来自配置层
        "message": "拍照识别还没接通，可以先把题目打进来。",
        "why": llm._vision_hint(),  # noqa: SLF001 提示语由配置层统一维护
    }


def _explain_api_error(raw: str) -> str:
    """把模型端的报错翻译成学生能懂的一句话。

    "400 · this model does not support image input" 这种原文直接甩到界面上，
    等于什么都没说。
    """
    low = raw.lower()
    if any(k in low for k in ("image", "vision", "multimodal", "multi-modal", "no mmproj")):
        return "当前模型不支持看图。换一个多模态模型，或者直接把题目打进来。"
    if "404" in low:
        return "这个地址上没有模型服务，检查一下端口和 /v1 有没有漏。"
    if "connect" in low or "refused" in low:
        return "本机模型服务没起来，先启动 LM Studio / Ollama。"
    return ""


def _sniff_mime(data: bytes, declared: str | None) -> str:
    """以文件头为准。手机上传时 content-type 常常是 application/octet-stream。"""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[4:12] in (b"ftypheic", b"ftypheix", b"ftyphevc", b"ftypmif1"):
        return "image/heic"
    return (declared or "").lower() or "image/png"


def extract(data: bytes, declared_mime: str | None = None) -> dict:
    if not data:
        return {"ok": False, "reason": "empty", "message": "这张图没读到内容，换一张试试？"}
    if len(data) > MAX_BYTES:
        return {"ok": False, "reason": "too_large",
                "message": "图片太大了，压缩一下或者截个图再传。"}

    # 先验输入再看模型：文件本身不对的话，接不接模型都白搭
    mime = _sniff_mime(data, declared_mime)
    if mime not in ALLOWED_MIME:
        return {"ok": False, "reason": "bad_type", "message": "只支持 png / jpg / webp 的图片。"}

    if not llm.vision_available():
        return unavailable()

    b64 = base64.b64encode(data).decode("ascii")
    try:
        raw = llm.chat_vision(SYSTEM, "把这道题抄成 LaTeX。", b64, mime)
    except llm.LLMTimeout as e:
        return {"ok": False, "reason": "timeout", "retryable": True,
                "message": f"模型还在算（{str(e)[:60]}），稍等一下再试一次；"
                           "本地模型首次加载会比较慢。"}
    except Exception as e:
        why = _explain_api_error(str(e))
        return {"ok": False, "reason": "api_error",
                "message": why or f"识别服务没连上（{str(e)[:80]}），可以先把题目打进来。"}

    data_json = llm.parse_json(raw) or {}
    latex = (data_json.get("latex") or "").strip()
    if not data_json.get("found", bool(latex)) or not latex:
        return {"ok": False, "reason": "not_found",
                "message": "这张图里没认出数学题，换个角度重拍，或者直接打进来。"}

    return {
        "ok": True,
        "latex": latex,
        "note": data_json.get("note", "") or "",
        "engine": f"vision · {llm._cfg.get('vision_model', '')}",  # noqa: SLF001 内部信息只做日志用途
    }
