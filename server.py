import os
import json
import asyncio
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_system import RAGSystem

app = FastAPI()
rag = RAGSystem()
CHAT_LOG_DIR = os.path.join(os.path.dirname(__file__), "chat_logs")
os.makedirs(CHAT_LOG_DIR, exist_ok=True)

# ✅ CORS 설정 (React 5173 포트에서 접근 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://44.208.140.83:5173",
	"http://medlinkbot.com",
        "https://medlinkbot.com",
        "http://www.medlinkbot.com",
        "https://www.medlinkbot.com"],  # 또는 ["*"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Health check
@app.get("/ping")
async def ping():
    return {"message": "pong"}

# ✅ 세션 목록 가져오기
@app.get("/api/list")
def list_sessions():
    return [f.replace(".json", "") for f in os.listdir(CHAT_LOG_DIR) if f.endswith(".json")]

# ✅ 세션 내용 불러오기
@app.get("/api/load")
def load_session(session_id: str):
    path = os.path.join(CHAT_LOG_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)

# ✅ 질문용 모델 (ask endpoint)
class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_medical_question(query: Query):
    try:
        answer = rag.query(query.question)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}

# ✅ 채팅용 POST (대화 기록 저장 포함)
@app.post("/api/chat")
def chat(request: Request):
    async def process():
        body = await request.json()
        question = body.get("question")
        session_id = body.get("session_id") or "default"
        user_id = body.get("user_id") or "web_user"

        answer = rag.run_graph(question, user_id)

        # 채팅 로그 저장
        log_path = os.path.join(CHAT_LOG_DIR, f"{session_id}.json")
        logs = []
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                logs = json.load(f)

        logs.append({"role": "user", "text": question})
        logs.append({"role": "bot", "text": answer["answer"]})

        with open(log_path, "w") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)

        return {"answer": answer["answer"]}
    return asyncio.run(process())
