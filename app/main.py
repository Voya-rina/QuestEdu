# -*- coding: utf-8 -*-
"""应用入口。

启动：
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
或（带局域网地址与二维码）：
    python scripts/run.py
"""
from __future__ import annotations

import sys

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import (
    routes_access,
    routes_llm,
    routes_meta,
    routes_ocr,
    routes_report,
    routes_session,
    routes_ws,
)
from app.config import APP_NAME, VERSION, WEB_DIR
from app.core.access import AccessGate

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def create_app() -> FastAPI:
    app = FastAPI(
        title=APP_NAME,
        version=VERSION,
        description="苏格拉底式追问 + 自评校准的数学伴学后端",
    )

    # 前端静态资源（含本地 vendor/KaTeX，不依赖任何 CDN）
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html", media_type="text/html; charset=utf-8")

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

    # 公网入口（隧道 / 托管）要过访问码；局域网直连不受影响。
    # 挂在 ASGI 层是因为 HTTP 中间件管不到 /ws/chat。
    app.add_middleware(AccessGate)
    return app


app = create_app()
