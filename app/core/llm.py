# -*- coding: utf-8 -*-
"""统一 LLM 适配层。

三种部署形态（对应大纲第三节），一行配置切换：
    mock    离线演示：不联网，走内置题库的确定性脚本
    cloud   云端 API（DeepSeek / OpenAI / 任意 OpenAI 兼容网关）
    local   本地 llama-server / Ollama（同样是 OpenAI 兼容接口）

配置优先级：进程环境变量 > 项目根目录 .env > 默认值。
页面上保存的配置会写进 .env（.env 已加进 .gitignore，不会被提交）。
"""
from __future__ import annotations

import json
import os
import re
import time

from app.config import ENV_PATH

_cfg: dict = {}
_last_usage: dict = {}
_last_raw: str = ""
_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def last_reply() -> str:
    """最近一次模型的原始回复。

    JSON 解析失败时上层拿它做最后一轮抢救——比如结构漏了 key，
    但 `"提问":"..."` 这些字段名还在，问句还是能捞出来的。
    """
    return _last_raw


class LLMError(RuntimeError):
    pass


class LLMTimeout(LLMError):
    """等模型回话等超时了。

    单独分类是因为它的处理方式和其他错误相反：重试没有意义（本地模型还在算，
    再发一次只会让它排更久的队），只能把等待时间说清楚。
    """


# ---------------------------------------------------------------- 配置

def _read_env() -> dict:
    out = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def norm_base(url: str) -> str:
    """规范化 base_url：补 /v1、去掉多余的斜杠。

    LM Studio / Ollama 用户手上拿到的地址通常是 `http://127.0.0.1:1234`，
    而 OpenAI 兼容路径在 `/v1` 下面。少写这一段的表现是 404，
    报错信息看起来像"模型没起"，其实只是路径拼接少了一层——所以这里兜住。
    """
    u = (url or "").strip().rstrip("/")
    if not u:
        return ""
    if u.endswith("/chat/completions"):
        u = u[: -len("/chat/completions")]
    if not u.endswith("/v1") and "/v1/" not in u:
        u += "/v1"
    return u


def reload() -> dict:
    """重新读取配置。保存 / 清除配置后调用。"""
    global _cfg
    env = _read_env()

    def pick(key: str, default: str = "") -> str:
        return (os.environ.get(key) or env.get(key) or default).strip()

    _cfg = {
        "mode": pick("INFERENCE_MODE").lower(),
        "base_url": norm_base(pick("LLM_BASE_URL", "https://api.deepseek.com/v1")),
        "api_key": pick("LLM_API_KEY"),
        "model": pick("LLM_MODEL", "deepseek-chat"),
        # 看图用的模型：留空表示"交给主模型试"，但本地模式下留空即视为不能看图
        "vision_model": pick("LLM_VISION_MODEL"),
        # 本地推理（LM Studio / llama-server / Ollama）：hybrid 模式下当云端不可用时兜底
        "local_base_url": norm_base(pick("LLM_LOCAL_BASE_URL", "http://127.0.0.1:1234/v1")),
        "local_model": pick("LLM_LOCAL_MODEL", "qwen3.8"),
        # 本地多模态模型（如 qwen2.5-vl-7b）。文本模型不吃图片，得单独指定
        "local_vision_model": pick("LLM_LOCAL_VISION_MODEL"),
        "local_api_key": pick("LLM_LOCAL_API_KEY"),
        # 云端通常十几秒；本地模型（尤其首次加载权重 + 思考型模型）慢一个量级。
        # 这两个来自同一个 bug：60 秒总超时把一次已经生成完的回答掐断了。
        "timeout": float(pick("LLM_TIMEOUT", "120") or 120),
        "local_timeout": float(pick("LLM_LOCAL_TIMEOUT", "900") or 900),
    }
    # 没显式指定模式时，有 key 就当在线，没有就当离线
    if not _cfg["mode"]:
        _cfg["mode"] = "cloud" if (_cfg["api_key"] and _cfg["base_url"]) else "mock"
    return _cfg


def save_config(base_url: str = "", api_key: str | None = None, model: str = "",
                vision_model: str | None = None,
                mode: str = "", local_base_url: str = "", local_model: str = "",
                local_vision_model: str | None = None,
                clear_key: bool = False) -> dict:
    """把配置写进 .env（清除时传空字符串）。

    - `mode` 为空表示按有没有 Key 自动判断（兼容旧调用）
    - `api_key` 传 None 表示**沿用已有 Key**：界面上那格是密码框，为了安全每次
      打开都是空的，输入框的空值不能当成"要删掉 Key"——否则改一次模型名就把
      Key 抹了，下次保存还得重贴。真要删用 `clear_key=True`。
    - 看图模型传 None 表示沿用 .env 里已有的值，避免一次保存把手工配的多模态模型抹掉
    - 本地模型（LM Studio / llama-server / Ollama）走 `LLM_LOCAL_*`，与云端配置并存，
      这样 hybrid 模式两条链路都留着
    """
    base_url = norm_base(base_url)
    model = (model or "").strip() or "deepseek-chat"
    mode = (mode or "").strip().lower()

    env = _read_env()
    if api_key is None:
        api_key = env.get("LLM_API_KEY", "")
    api_key = (api_key or "").strip()
    if clear_key:
        api_key = ""
    if vision_model is None:
        vision_model = env.get("LLM_VISION_MODEL", "")
    if local_vision_model is None:
        local_vision_model = env.get("LLM_LOCAL_VISION_MODEL", "")
    local_base_url = norm_base(
        (local_base_url or "").strip() or env.get("LLM_LOCAL_BASE_URL", "http://127.0.0.1:1234/v1"))
    local_model = (local_model or "").strip() or env.get("LLM_LOCAL_MODEL", "qwen3.8")

    if mode not in ("mock", "cloud", "local", "hybrid"):
        mode = "cloud" if (api_key and base_url) else "mock"

    body = (
        "# QuestEdu 本地配置。含密钥，已被 .gitignore 排除，不要提交。\n"
        f"INFERENCE_MODE={mode}\n"
        f"LLM_BASE_URL={base_url or 'https://api.deepseek.com/v1'}\n"
        f"LLM_API_KEY={api_key}\n"
        f"LLM_MODEL={model}\n"
        f"LLM_VISION_MODEL={vision_model.strip()}\n"
        f"LLM_LOCAL_BASE_URL={local_base_url}\n"
        f"LLM_LOCAL_MODEL={local_model}\n"
        f"LLM_LOCAL_VISION_MODEL={local_vision_model.strip()}\n"
    )
    ENV_PATH.write_text(body, encoding="utf-8")
    return reload()


def mode() -> str:
    return _cfg.get("mode", "mock")


def available() -> bool:
    """有没有可用的大模型端点（云端或本地，看模式）。"""
    return bool(_routes())


def online() -> bool:
    """是否走"真模型"路径（对应大纲的 cloud / local / hybrid 三模式）。

    mock 模式下引擎用内置脚本判定；其余三种都该让模型参与。
    """
    return available()


def vision_available() -> bool:
    """能不能看图。"""
    return bool(vision_routes())


def vision_routes() -> list[dict]:
    """看图请求该发给谁。

    规则（这几个分支正好对应用户踩到的坑）：
    - 云端：显式配了看图模型就用它，没配就拿主模型试一次——有些网关的模型本身吃图
    - 本地：**必须显式配多模态模型**。LM Studio 里跑的 qwen3.8 是文本模型，
      把图片塞过去只会拿到 400，而且要等它慢慢处理完才报错
    - 全都没有：返回空，上层明确告诉学生"拍照没接通"，而不是让他等两分钟
    """
    out: list[dict] = []
    for r in _routes():
        if r["name"] == "cloud":
            model = (_cfg.get("vision_model") or "").strip() or r["model"]
            out.append({**r, "model": model})
        else:
            model = (_cfg.get("local_vision_model") or "").strip()
            if model:
                out.append({**r, "model": model})
    return out


def _routes() -> list[dict]:
    """按模式排出候选端点，前一个失败就试后一个。

    mock   → 空（离线脚本兜底）
    cloud  → 云端
    local  → 本机 LM Studio / llama-server / Ollama
    hybrid → 云端优先，不通则回落本地
    """
    m = mode()
    out: list[dict] = []
    if m in ("cloud", "hybrid") and _cfg.get("base_url") and _cfg.get("api_key"):
        out.append({
            "name": "cloud", "base_url": _cfg["base_url"],
            "api_key": _cfg["api_key"], "model": _cfg["model"],
            "timeout": _cfg["timeout"],
        })
    if m in ("local", "hybrid") and _cfg.get("local_base_url"):
        # 本地服务通常不校验 Key，但 OpenAI 兼容协议要求头里有个东西
        out.append({
            "name": "local", "base_url": _cfg["local_base_url"],
            "api_key": _cfg.get("local_api_key") or "sk-local",
            "model": _cfg["local_model"],
            "timeout": _cfg["local_timeout"],
        })
    return out


def active_model() -> str:
    """当前实际会用的模型名（前端提示、自检回显都该用它，而不是云端那个）。"""
    rs = _routes()
    return rs[0]["model"] if rs else _cfg.get("model", "")


def describe() -> dict:
    vr = vision_routes()
    return {
        "mode": mode(),
        "ready": available(),
        "routes": [r["name"] for r in _routes()],
        "base_url": _cfg.get("base_url"),
        "model": _cfg.get("model"),
        "active_model": active_model(),
        "local_base_url": _cfg.get("local_base_url"),
        "local_model": _cfg.get("local_model"),
        "vision_model": _cfg.get("vision_model"),
        "local_vision_model": _cfg.get("local_vision_model"),
        "vision_ready": bool(vr),
        "vision_target": (f"{vr[0]['model']} @ {vr[0]['name']}" if vr else ""),
        # 不能看图时，把"为什么"和"怎么改"直接说给界面，别让用户自己猜
        "vision_hint": "" if vr else _vision_hint(),
        "key_configured": bool(_cfg.get("api_key")),
        "key_hint": (_cfg.get("api_key", "")[:6] + "…") if _cfg.get("api_key") else "",
        "local_timeout": int(_cfg.get("local_timeout", 900)),
        "env_file": str(ENV_PATH),
    }


def _vision_hint() -> str:
    if not available():
        return "还没有接入大模型"
    if mode() == "local" or (mode() == "hybrid" and not _cfg.get("api_key")):
        return "本地模型不能看图，需要另配一个多模态模型（如 qwen2.5-vl-7b）"
    return "当前模型可能不支持看图，配一个多模态模型名更稳妥"


reload()


# ---------------------------------------------------------------- 调用

def _strip_fence(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def chat(system: str, user: str, temperature: float = 0.3) -> str:
    """纯文本对话。"""
    return _post(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature,
    )


def chat_vision(system: str, user: str, image_b64: str, mime: str = "image/png",
                temperature: float = 0.1) -> str:
    """带一张图的对话（拍照 OCR 用）。

    走 OpenAI 兼容的 `image_url` + data URL 形式，DeepSeek-VL / Qwen-VL / GPT-4o /
    本地 llama-server 的多模态模型都是这套格式。
    """
    content = [
        {"type": "text", "text": user},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
    ]
    return _post(
        [{"role": "system", "content": system}, {"role": "user", "content": content}],
        temperature,
        routes=vision_routes(),      # 注意：不是 _routes()，看图要命中多模态模型
    )


def _post(messages: list, temperature: float, routes: list[dict] | None = None) -> str:
    routes = _routes() if routes is None else routes
    if not routes:
        raise LLMError("未配置 LLM（当前模式 %s）" % mode())

    errors: list[str] = []
    all_timeout = True
    for route in routes:
        try:
            out = _send(route, messages, temperature)
            globals()["_last_raw"] = out
            return out
        except LLMTimeout as e:
            errors.append(f"{route['name']} · {e}")
        except LLMError as e:
            all_timeout = False
            errors.append(f"{route['name']} · {e}")
    msg = "; ".join(errors)
    if all_timeout:
        raise LLMTimeout(msg)
    # hybrid 下两个都挂了，把两条错误都带出来，方便判断是 Key 问题还是本地没起
    raise LLMError(msg)


def _send(route: dict, messages: list, temperature: float) -> str:
    import httpx  # 延迟导入：离线模式完全不需要

    headers = {"Content-Type": "application/json"}
    headers["Authorization"] = f"Bearer {route['api_key']}"
    payload = {
        "model": route["model"],
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    # connect 单独设小：地址填错要立刻报错，不能陪着 read timeout 一起等十几分钟
    limit = httpx.Timeout(connect=8, read=route.get("timeout") or 120, write=60, pool=8)
    t0 = time.time()
    try:
        with httpx.Client(timeout=limit) as client:
            r = client.post(f"{route['base_url']}/chat/completions", headers=headers, json=payload)
    except httpx.TimeoutException:
        raise LLMTimeout(
            f"等了 {int(time.time() - t0)} 秒还没回话"
            + ("（本地模型首次加载权重可能要一两分钟）" if route["name"] == "local" else "")
        )
    except httpx.HTTPError as e:
        raise LLMError(f"连不上 {route['base_url']}（{type(e).__name__}: {e}）")

    if r.status_code >= 400:
        detail = r.text[:300]
        try:
            j = r.json()
            msg = (j.get("error") or {}).get("message") or j.get("message")
            if msg:
                detail = msg
        except Exception:
            pass
        raise LLMError(f"HTTP {r.status_code} · {detail}")

    try:
        body = r.json()
        choice = body["choices"][0]
    except Exception:
        raise LLMError(f"返回格式不认识：{r.text[:200]}")
    _last_usage.clear()
    _last_usage.update(body.get("usage") or {})
    return content_of(choice)


def content_of(choice: dict) -> str:
    """从一条 choice 里取回复正文。

    思考型模型（Qwen3 / DeepSeek-R1 系）会把过程写在 `reasoning_content`、
    `content` 留空。LM Studio 上尤其常见——按"空回复"处理会白白丢掉一次
    已经算完的推理。实测日志里就是 `content: ""` 配 `reasoning_content: "..."`。
    """
    msg = choice.get("message") or {}
    content = (msg.get("content") or "").strip()
    if not content:
        content = (msg.get("reasoning_content") or "").strip()
    if not content:
        raise LLMError(f"模型返回了空内容（finish_reason={choice.get('finish_reason')}）")
    return content


# JSON 里 `\` 后面的一个字符（含 \uXXXX 与已转义的 \" \\ \/）
_ESCAPE = re.compile(r'\\(u[0-9a-fA-F]{4}|["\\/]|.)', re.S)


def _fix_escape(m: re.Match) -> str:
    body = m.group(1)
    if body[0] == "u" and len(body) == 5:
        return m.group(0)               # \u4e2d 这类 unicode 转义，保留
    if body in ('"', "\\", "/", "n"):
        # 引号、反斜杠、斜杠是写对了；\n 放行——模型确实用它分段，
        # 代价是 `\neq` 会被当成换行，比把每处 \n 都当字面量划算
        return m.group(0)
    return "\\\\" + body                # 其余当 LaTeX 命令，补一个反斜杠


def repair_json(text: str) -> str:
    r"""修掉 JSON 里没转义的 LaTeX 反斜杠。

    模型会自然地写 `"期望答案": "$\frac{1}{2}$"` 这种原样形式。其中 `\f` 恰好
    是合法 JSON 转义（换页符），`\lim` 里的 `\l` 不是——整段 JSON 在第一个
    `\l` 处就解析失败。实测 Gemma 4 每次带 `\to` 都会踩。

    做法：`\` 后面不是合法转义就补一个，让它在字符串里成为字面的反斜杠。
    注意 `\f` `\t` `\n` 这几个"合法的"也要补——模型写的是 `\frac`、`\tan`、
    `\neq`，不补就会变成换页符、制表符、换行符，答题界面上直接乱码。
    已经写对的 `\\frac` 不受影响（第一个 `\` 后面跟的是 `\`，走保留分支）。
    """
    return _ESCAPE.sub(_fix_escape, text)


def parse_json(raw: str) -> dict | None:
    """从模型输出里抠出 JSON：先修 LaTeX 转义，再整体解析，最后退回第一个 `{...}` 块。

    顺序很重要：`"$\frac{1}{2}$"` 里 `\f` 是**合法** JSON 转义，直接 json.loads
    会"成功"但把 `\frac` 变成换页符——不报错，静默坏掉，更难查。所以先修后解析。
    """
    text = _strip_fence(raw)
    for candidate in (repair_json(text), text):
        try:
            return json.loads(candidate)
        except Exception:
            continue
    m = _JSON_BLOCK.search(repair_json(text))
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            return None
    return None


def chat_json(system: str, user: str, retries: int | None = None, temperature: float = 0.2) -> dict:
    """强制 JSON 输出。解析失败自动重试（大纲：结构化输出约束）。

    但**本地只重试一次**，超时则完全不重试：
    - 本地模型一次推理要几分钟，重发只会让它在队列里排更久（LM Studio 日志里的
      "Client disconnected" 就是这么来的——客户端走了，服务端还在算）
    - 解析失败重试三次，在本地就意味着多等十来分钟，学生早走了
    """
    if retries is None:
        retries = 1 if mode() == "local" else 3
    last = ""
    for attempt in range(retries):
        prompt = system
        if attempt:
            prompt = system + ("\n\n上一次输出不是合法 JSON。只输出 JSON 对象本身，"
                               "不要 Markdown 代码块、不要任何解释文字。")
        raw = chat(prompt, user, temperature=temperature)
        data = parse_json(raw)
        if data is not None:
            return data
        last = "无法解析为 JSON 对象"
    raise LLMError(f"JSON 解析连续失败：{last}")


def list_models() -> list[dict]:
    """列出各端点可用的模型名。

    本地模型名经常手打出错——LM Studio 里显示成 `gemma 4`，id 里就真的带空格。
    与其让人抄，不如直接列出来让他选。
    任何一路失败都只是跳过，不能因为模型没起就打不开设置面板。
    """
    import httpx

    out: list[dict] = []
    for route in _routes():
        try:
            with httpx.Client(timeout=httpx.Timeout(connect=3, read=8, write=5, pool=3)) as c:
                resp = c.get(f"{route['base_url']}/models",
                             headers={"Authorization": f"Bearer {route['api_key']}"})
            if resp.status_code >= 400:
                continue
            for m in (resp.json().get("data") or []):
                mid = m.get("id")
                if mid:
                    out.append({"id": mid, "route": route["name"]})
        except Exception:
            continue
    return out


def test() -> dict:
    """连通性自检：发一个极短的请求，报告延迟与回显。

    顺带量一下生成速度——本地模型慢是主要矛盾（实测 2~5 t/s，
    一道题要拆 500~1500 token），这个数决定了一次拆题要等多久。
    """
    if not available():
        return {"ok": False, "message": "还没有接入模型"}
    _last_usage.clear()
    t0 = time.time()
    try:
        out = chat("你是一个只回 JSON 的接口测试器。",
                   '返回 {"ok":true}', temperature=0)
        sec = time.time() - t0
        used = active_model()
        tokens = int(_last_usage.get("completion_tokens") or 0)
        out_dict = {
            "ok": True,
            "latency_ms": int(sec * 1000),
            "model": used,
            "echo": out[:80],
            "completion_tokens": tokens,
            "vision_ready": vision_available(),
            "vision_hint": "" if vision_available() else _vision_hint(),
        }
        # 样本太小时算不出可信的速率，宁愿不显示
        if tokens >= 10 and sec > 0:
            out_dict["tokens_per_sec"] = round(tokens / sec, 1)
        return out_dict
    except LLMTimeout as e:
        return {"ok": False, "message": str(e)[:300],
                "latency_ms": int((time.time() - t0) * 1000),
                "timeout": True}
    except Exception as e:
        return {"ok": False, "message": str(e)[:300],
                "latency_ms": int((time.time() - t0) * 1000)}
