#!/usr/bin/env bash
cd "$(dirname "$0")"
PORT="${1:-3070}"
echo "MyAI ishga tushmoqda -> http://localhost:$PORT"
exec ./venv/bin/python server.py "$PORT"
