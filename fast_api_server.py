from flask import Flask, request, jsonify
from rag_system import RAGSystem
from flask_cors import CORS
import os
import json
from datetime import datetime
import feedparser
import uvicorn

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

rag = RAGSystem()
CHAT_LOG_DIR = "chat_logs"
os.makedirs(CHAT_LOG_DIR, exist_ok=True)

# 뉴스 
@app.route("/api/news", methods=["GET"])
def get_news():
    try:
        rss_url = "https://www.medicaldaily.com/rss"
        feed = feedparser.parse(rss_url)

        items = []
        for entry in feed.entries[:8]:
            title = entry.title
            link = entry.link if isinstance(entry.link, str) else str(entry.link)

            items.append({
                "title": title,
                "link": link
            })
        return jsonify({"items": items})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ 채팅 저장 함수
def save_chat(user_id, question, answer):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    session_id = f"{user_id}_{timestamp}"
    filepath = os.path.join(CHAT_LOG_DIR, f"{session_id}.json")
    chat_data = [{"role": "user", "text": question}, {"role": "bot", "text": answer}]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(chat_data, f, ensure_ascii=False, indent=2)

# ✅ 채팅 목록 반환
@app.route("/api/list", methods=["GET"])
def list_sessions():
    files = sorted(
        [f for f in os.listdir(CHAT_LOG_DIR) if f.endswith(".json")],
        reverse=True
    )
    sessions = [f.replace(".json", "") for f in files]
    return jsonify(sessions)

# ✅ 특정 세션 불러오기
@app.route("/api/load", methods=["GET"])
def load_session():
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400
    filepath = os.path.join(CHAT_LOG_DIR, f"{session_id}.json")
    if not os.path.exists(filepath):
        return jsonify({"error": "Session not found"}), 404
    with open(filepath, "r", encoding="utf-8") as f:
        chat_data = json.load(f)
    return jsonify(chat_data)

# ✅ 챗봇 응답 및 저장
@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        question = data.get("question", "")
        user_id = data.get("user_id", "guest")
        result = rag.run_graph(question, user_id)
        answer = result.get("answer", "❌ 답변 생성 실패") if isinstance(result, dict) else str(result)

        # ✅ 저장용 세션 ID 생성
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        session_id = f"{user_id}_{timestamp}"
        filepath = os.path.join(CHAT_LOG_DIR, f"{session_id}.json")

        chat_data = [{"role": "user", "text": question}, {"role": "bot", "text": answer}]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(chat_data, f, ensure_ascii=False, indent=2)

        return jsonify({"answer": answer, "session_id": session_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, threaded=True) # 서버재시작 없이 반영하려고 8000에서 8001로 변경
