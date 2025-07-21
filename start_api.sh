#!/bin/bash

source ~/Medical-Chatbot/venv/bin/activate
cd ~/Medical-Chatbot

# ✅ 이미 8000 포트가 떠 있으면 실행하지 않음
if lsof -i :8000 > /dev/null
then
  echo "⚠️ 8000 포트가 이미 사용 중입니다. uvicorn 실행하지 않음."
  exit 1
fi

# 🔥 exec로 uvicorn을 메인 프로세스로 실행 (PM2가 이걸 감시함)
exec uvicorn server:app --host 0.0.0.0 --port 8000

