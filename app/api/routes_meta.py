# -*- coding: utf-8 -*-
"""元信息与健康检查。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.config import APP_NAME, VERSION, lan_ip
from app.core import access, console_data, llm
from app.core.store import store
from app.data.problems import PROBLEMS

router = APIRouter(tags=["meta"])


def _port(request: Request) -> int:
    """当前实际监听的端口。

    不能写死 8000：启动器会把端口往后顺延（8062 起），二维码和局域网地址
    都得跟着走，否则手机上扫码打开的是个空页面。
    """
    return request.url.port or 80


def problems_brief() -> list[dict]:
    """下发给前端的题目摘要：不带答案、不带节点内容。"""
    return [
        {
            "id": p["id"],
            "category": p.get("category", "其他"),
            "题型": p["题型"],
            "latex": p["latex"],
            "难度": p["难度"],
            "知识点": p["知识点"],
            "steps": len(p["nodes"]),
        }
        for p in PROBLEMS
    ]


@router.get("/api/info")
def info(request: Request):
    # 界面按「这台设备实际能用什么」渲染：MOCK 权限的设备一律看到离线题库——
    # 就算服务端此刻跑着本地模型，自定义题和拍照也不给它出现
    xff = request.headers.get("x-forwarded-for", "")
    peer = access.normalize_peer(xff.split(",")[0].strip() if xff
                                 else (request.client.host if request.client else ""))
    is_ctrl = access.is_controller(peer, request.headers)
    llm_desc = llm.describe(full=False)
    if not is_ctrl and console_data.device_perm(peer) == "mock":
        llm_desc = {
            "mode": "mock", "ready": False, "vision_ready": False,
            "vision_hint": "本设备当前是离线题库权限（MOCK）",
            "local_timeout": llm_desc.get("local_timeout", 900),
        }
    return {
        "app": APP_NAME,
        "version": VERSION,
        # 用户端可见的最小集：配置细节（Key 提示、env 路径、端点地址）不下发
        "llm": llm_desc,
        "demo_problem": PROBLEMS[0]["latex"],
        "problems": problems_brief(),
        "lan_url": f"http://{lan_ip()}:{_port(request)}",
    }


@router.get("/api/problems")
def list_problems():
    return {"problems": problems_brief()}


@router.get("/api/health")
def health():
    return {"ok": True, "sessions": store.count(), "mode": llm.mode(), "version": VERSION}


@router.get("/api/qr")
def qr(request: Request, u: str = ""):
    """入口二维码（SVG）。

    答辩现场把手机递给评委之前，先让他扫这一下——比手输 IP 靠谱。
    `?u=` 可以指定要编码的地址（IPv6 带方括号、隧道域名都行）；
    只允许自家地址，否则就是个免费的二维码生成接口了。
    """
    import io

    from fastapi.responses import Response

    from app.core.access import safe_qr_target

    try:
        import qrcode
        import qrcode.image.svg
    except ImportError:
        raise HTTPException(501, "没装 qrcode：pip install qrcode")

    url = (u or "").strip() or f"http://{lan_ip()}:{_port(request)}"
    if not safe_qr_target(url):
        raise HTTPException(400, "只能给本站地址生成二维码")

    img = qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage)
    buf = io.BytesIO()
    img.save(buf)
    return Response(buf.getvalue(), media_type="image/svg+xml",
                    headers={"Cache-Control": "no-store"})
