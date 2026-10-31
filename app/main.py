# -*- coding: utf-8 -*-
"""应用入口。

启动：
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
或（带局域网地址与二维码）：
    python scripts/run.py
"""
from __future__ import annotations

import logging
import sys

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api import (
    routes_access,
    routes_console,
    routes_llm,
    routes_meta,
    routes_ocr,
    routes_report,
    routes_session,
    routes_ws,
)
from app.config import APP_NAME, DATA_DIR, VERSION, WEB_DIR
from app.core.access import AccessGate, is_controller
from app.core.console_data import ring_log

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 终端业务日志：OCR 收图/结果、每次 LLM 调用的路由与耗时、引擎判定、
# 自评与变式，全部打到 stdout（START.cmd 的窗口里直接看），问题定位不靠猜。
# 同时写一份到 data/questedu.log（1MB 滚动）——控制台窗口滚没了/关了也能查。
# uvicorn 自己的 access log 照旧，两层互不干扰。
def _setup_logging() -> None:
    root = logging.getLogger()
    if root.handlers:            # 重复导入（测试/热重载）时不叠加 handler
        return
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                            "%H:%M:%S")
    root.setLevel(logging.INFO)

    cons = logging.StreamHandler()
    cons.setFormatter(fmt)
    root.addHandler(cons)
    ring_log.setFormatter(fmt)
    root.addHandler(ring_log)
    try:
        from logging.handlers import RotatingFileHandler

        fh = RotatingFileHandler(DATA_DIR / "questedu.log",
                                 maxBytes=1_000_000, backupCount=2, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except Exception:
        pass                     # 日志文件写不了不影响服务
    # httpx 对每个请求都吐一行 INFO，和 uvicorn 的 access log 重复——降噪
    logging.getLogger("httpx").setLevel(logging.WARNING)


_setup_logging()


def create_app() -> FastAPI:
    app = FastAPI(
        title=APP_NAME,
        version=VERSION,
        description="苏格拉底式追问 + 自评校准的数学伴学后端",
    )

    # 前端静态资源（含本地 vendor/KaTeX，不依赖任何 CDN）
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    # ---------- 页面路由：按角色分流 ----------
    # `/` 是角色分流入口：控制端（跑服务的电脑）打开 → 控制台 Dashboard；
    # 扫码进来的用户端打开 → 学生页。角色判定永远在服务端（is_controller）。
    @app.get("/", include_in_schema=False)
    def index(request: Request):
        peer = request.client.host if request.client else ""
        if is_controller(peer, request.headers):
            return FileResponse(WEB_DIR / "console.html",
                                media_type="text/html; charset=utf-8")
        return FileResponse(WEB_DIR / "index.html",
                            media_type="text/html; charset=utf-8")

    # 学生页固定地址：控制端也能打开预览学生视图
    @app.get("/student", include_in_schema=False)
    def student():
        return FileResponse(WEB_DIR / "index.html",
                            media_type="text/html; charset=utf-8")

    # 控制台固定地址：非控制端访问一律弹回学生页（防 URL 猜测）
    @app.get("/console", include_in_schema=False)
    def console_page(request: Request):
        peer = request.client.host if request.client else ""
        if not is_controller(peer, request.headers):
            return RedirectResponse("/", status_code=302)
        return FileResponse(WEB_DIR / "console.html",
                            media_type="text/html; charset=utf-8")

    # PWA 的两个文件必须挂在根路径：manifest 的 scope 与 service worker 的
    # 控制范围都由它们自己的 URL 决定，放在 /static/ 下就只能控制 /static/
    @app.get("/manifest.json", include_in_schema=False)
    def manifest():
        return FileResponse(WEB_DIR / "manifest.json",
                            media_type="application/manifest+json")

    @app.get("/sw.js", include_in_schema=False)
    def service_worker():
        return FileResponse(
            WEB_DIR / "sw.js",
            media_type="application/javascript",
            headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"},
        )

    app.include_router(routes_meta.router)
    app.include_router(routes_session.router)
    app.include_router(routes_report.router)
    app.include_router(routes_ocr.router)
    app.include_router(routes_ws.router)
    app.include_router(routes_llm.router)
    app.include_router(routes_access.router)
    app.include_router(routes_console.router)

    # 公网入口（隧道 / 托管）要过访问码；局域网直连不受影响。
    # 挂在 ASGI 层是因为 HTTP 中间件管不到 /ws/chat。
    app.add_middleware(AccessGate)
    return app


app = create_app()
