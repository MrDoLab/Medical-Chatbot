# ✅ 목표: 기존 `main.py` CLI 콘솔 기반 챗봇에 FastAPI 백엔드 기능 추가하기

# ✅ 1단계: `main.py` 기능을 FastAPI 서버로 분리
# 새 파일: `api.py` 또는 `server.py` 생성

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from rag_system import RAGSystem
from pathlib import Path
import os, json

app = FastAPI()
rag_system = RAGSystem()
CHAT_LOG_DIR = "chat_logs"
os.makedirs(CHAT_LOG_DIR, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/list")
def list_sessions():
    return [f.replace(".json", "") for f in os.listdir(CHAT_LOG_DIR) if f.endswith(".json")]

@app.get("/api/load")
def load_session(session_id: str):
    path = os.path.join(CHAT_LOG_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)

@app.post("/api/chat")
def chat(request: Request):
    import asyncio
    async def process():
        body = await request.json()
        question = body.get("question")
        session_id = body.get("session_id") or "default"
        user_id = body.get("user_id") or "web_user"

        answer = rag_system.run_graph(question, user_id)

        # 저장
        log_path = os.path.join(CHAT_LOG_DIR, f"{session_id}.json")
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append({"user": question, "bot": answer["answer"]})
        with open(log_path, "w") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)

        return {"answer": answer["answer"]}
    return asyncio.run(process())

# ✅ 실행 명령:
# uvicorn server:app --host 0.0.0.0 --port 8000

# ✅ 기존 main.py는 CLI 용으로 따로 두되, 서버용은 이걸로 사용
