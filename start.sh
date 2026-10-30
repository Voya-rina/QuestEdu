#!/usr/bin/env bash
# QuestEdu —— macOS / Linux 一键启动
# 后端与页面由同一个进程提供，起一次就都有了。
set -e
cd "$(dirname "$0")"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    if [ -x ".venv/bin/python" ]; then
        PY=".venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PY="python3"
    elif command -v python >/dev/null 2>&1; then
        PY="python"
    else
        echo "没找到 Python，先装 3.10+：https://www.python.org/downloads/"
        exit 1
    fi
fi

exec "$PY" scripts/run.py "$@"
