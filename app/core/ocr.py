# -*- coding: utf-8 -*-
"""拍照 → LaTeX（大纲第三节第 1 条）。

云端走多模态模型，端侧以后可以换 OXR-1.0；接口形状一致，换引擎只改这一层。

铁律：**识别不出来就说识别不出来**。不要把模型瞎猜的式子当成学生的题目送进
追问流程——那会让学生在"一道自己没拍的题"上答题，比识别失败糟糕得多。
"""
from __future__ import annotations

import base64
import json
import logging
import re
import time

from app.core import llm

log = logging.getLogger(__name__)

MAX_BYTES = 8 * 1024 * 1024          # 8MB，手机直出照片通常 2~4MB
ALLOWED_MIME = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/heic", "image/heif"}

# 纯文本回退用：教材题号前缀（"3. "）、拒绝话术
_NUM_PREFIX = re.compile(r"^\s*\d+\s*[.、)）:：]\s*")
_REFUSAL = re.compile(r"无法识别|看不清|不能识别|抱歉|sorry|cannot", re.I)
# 小模型照抄提示词模板的 note 特征
_NOTE_ECHO = re.compile(r"没有就|题目里的文字要求|空串")
# 纯文本回退要挡掉的 JSON 特征：整段 {"found":...} 被当题面塞给学生的丑态绝不能再出现
_JSON_LOOK = re.compile(r'^\s*\{|\s*"(?:found|latex|note)"\s*:')
# 兜底抠字段：JSON 整体解析失败但 latex 字符串本身写对了，直接按引号配对抠出来
_LATEX_FIELD = re.compile(r'"latex"\s*:\s*"((?:[^"\\]|\\.)*)"', re.S)


def _salvage_latex(raw: str) -> str:
    """JSON 解析失败时的最后抢救：正则抠出 "latex" 字段再反转义。

    parse_json 失败多半是 JSON 别处坏了（控制字符、截断），但 latex 那个
    字符串本身可能是完好的——直接丢掉太可惜。先过 repair_json 把 LaTeX
    反斜杠修对，再反转义；抠不到就空串，绝不把 JSON 原文当题面。
    """
    m = _LATEX_FIELD.search(llm.repair_json(raw or ""))
    if not m:
        return ""
    try:
        return json.loads('"' + m.group(1) + '"').strip()
    except Exception:
        return m.group(1).strip()


def _plain_transcription(raw: str) -> str:
    """专用 OCR 模型（minicpm-ocr 等）不理会 JSON 指令，直接吐纯文本转写。

    文本像一道题（有实际内容、不是拒绝话术）就整段当题面用——识别结果
    反正要学生过目确认，宁可用上，别白白报"没认出数学题"。
    拒绝话术 / 太短 / 太长 / 长得像 JSON 的仍然当失败——像 JSON 说明
    parse 环节已经失败过一次，把原文塞给学生只会看到一坨大括号。
    """
    t = (raw or "").strip()
    t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t).strip()   # 万一包了代码块
    t = _NUM_PREFIX.sub("", t)                                # "3. 求函数…" → "求函数…"
    t = re.sub(r"\s+", " ", t)
    if not (4 <= len(t) <= 500) or _REFUSAL.search(t) or _JSON_LOOK.search(t):
        return ""
    return t


def _latex_delims(s: str) -> str:
    """模型爱用 \\(...\\) / \\[...\\] 写行内/行间公式；统一换成 $ / $$，
    前端 pbar 的"没有 $ 就整段包 $$"逻辑才不会把它们再包一层弄坏渲染。"""
    s = re.sub(r"\\\((.+?)\\\)", r"$\1$", s, flags=re.S)
    s = re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", s, flags=re.S)
    return s

SYSTEM = """你是题目转写器。把图片里的数学题抄成 LaTeX，不做任何解答。

只输出 JSON，三个字段：
{"found":true,"latex":"...","note":"..."}

字段说明：
- latex：题目的 LaTeX，行内公式用 $...$
- note：题目文字要求的原文（例："求 x=1 处的导数"），没有就填 ""

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
        log.info("OCR 拒绝：没有可用的看图模型（%s）", llm._vision_hint())
        return unavailable()

    b64 = base64.b64encode(data).decode("ascii")
    targets = [f"{r['model']}@{r['name']}" for r in llm.vision_routes()]
    t0 = time.time()
    log.info("OCR 收图 %d 字节 · mime=%s · 目标=%s", len(data), mime, ",".join(targets))
    try:
        raw = llm.chat_vision(SYSTEM, "把这道题抄成 LaTeX。", b64, mime)
    except llm.LLMTimeout as e:
        log.warning("OCR 超时（%dms）：%s", int((time.time() - t0) * 1000), str(e)[:120])
        return {"ok": False, "reason": "timeout", "retryable": True,
                "message": f"模型还在算（{str(e)[:60]}），稍等一下再试一次；"
                           "本地模型首次加载会比较慢。"}
    except Exception as e:
        log.warning("OCR 调用失败（%dms）：%s", int((time.time() - t0) * 1000), str(e)[:200])
        why = _explain_api_error(str(e))
        return {"ok": False, "reason": "api_error",
                "message": why or f"识别服务没连上（{str(e)[:80]}），可以先把题目打进来。"}

    data_json = llm.parse_json(raw) or {}
    latex = (data_json.get("latex") or "").strip()
    if not latex and not data_json:
        # JSON 没解析出来：先试着从坏 JSON 里直接抠 latex 字段，
        # 抠不到且整段像纯文本转写才用纯文本；都不行就当失败，
        # 坚决不把 {"found":true,...} 这类原文塞进题面
        latex = _salvage_latex(raw) or _plain_transcription(raw)
        if latex:
            log.info("OCR 非 JSON 输出，%s采用",
                     "从 latex 字段抢救" if _LATEX_FIELD.search(raw) else "按纯文本转写")
    latex = _latex_delims(latex) if latex else latex
    if not data_json.get("found", bool(latex)) or not latex:
        log.info("OCR 未认出数学题（%dms）· 模型原话开头：%s",
                 int((time.time() - t0) * 1000), raw[:80].replace("\n", " "))
        return {"ok": False, "reason": "not_found",
                "message": "这张图里没认出数学题，换个角度重拍，或者直接打进来。"}

    used = llm._last_route  # noqa: SLF001  仅日志用途
    # 小模型会把提示词里的字段说明原样抄回 note（"题目里的文字要求(没有就空串)"）
    note = data_json.get("note", "") or ""
    if _NOTE_ECHO.search(note):
        note = ""
    log.info("OCR 成功 · %s@%s · %dms · latex %d 字",
             used.get("model", "?"), used.get("name", "?"),
             int((time.time() - t0) * 1000), len(latex))
    return {
        "ok": True,
        "latex": latex,
        "note": note,
        # 如实上报实际干活的那个端点：本地 minicpm-ocr 识别时，
        # 以前这里写死云端 vision_model，排查"为什么没用本地模型"会被带偏
        "engine": "vision · {model} @ {name}".format(
            model=(llm._last_route.get("model") or "?"),  # noqa: SLF001
            name=(llm._last_route.get("name") or "?"),    # noqa: SLF001
        ),
    }
