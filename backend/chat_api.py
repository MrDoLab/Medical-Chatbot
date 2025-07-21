# backend/chat_api.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os, json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # CORS: React와 연결 위해 허용
    allow_methods=["*"],
    allow_headers=["*"]
)

LOG_DIR = "../chat_logs"
os.makedirs(LOG_DIR, exist_ok=True)

@app.post("/api/save")
async def save_chat(req: Request):
    data = await req.json()
    session_id = data["session_id"]
    filepath = os.path.join(LOG_DIR, f"{session_id}.json")

    history = []
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            history = json.load(f)

    history.append({"user": data["user"], "bot": data["bot"]})
    with open(filepath, "w") as f:
        json.dump(history, f)
    return {"status": "saved"}

@app.get("/api/list")
def list_sessions():
    return [f.replace(".json", "") for f in os.listdir(LOG_DIR) if f.endswith(".json")]

@app.get("/api/load")
def load_session(session_id: str):
    path = os.path.join(LOG_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)
