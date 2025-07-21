import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_system import RAGSystem
from datetime import datetime

# app = Flask(__name__)
# CORS(app)

# rag = RAGSystem()

CONVO_DIR = "conversations"
os.makedirs(CONVO_DIR, exist_ok=True)


def save_message(user_id, question, answer):
    filename = os.path.join(CONVO_DIR, f"{user_id}.json")
    data = []
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    timestamp = datetime.now().isoformat()
    if data and "messages" in data[-1] and not data[-1].get("completed", False):
        # 기존 대화 이어쓰기
        data[-1]["messages"].append({"role": "user", "text": question})
        data[-1]["messages"].append({"role": "bot", "text": answer})
    else:
        # 새 대화
        data.append({
            "id": f"{user_id}_{len(data)+1}",
            "title": question[:20],  # 대화 제목
            "messages": [
                {"role": "user", "text": question},
                {"role": "bot", "text": answer}
            ],
            "timestamp": timestamp,
            "completed": False
        })
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.json
        question = data.get("question", "")
        user_id = data.get("user_id", "user_default")

        result = rag.run_graph(question, user_id)
        answer = result.get("answer") if isinstance(result, dict) else str(result)

        save_message(user_id, question, answer)

        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history/<user_id>", methods=["GET"])
def get_conversation_titles(user_id):
    filename = os.path.join(CONVO_DIR, f"{user_id}.json")
    if not os.path.exists(filename):
        return jsonify([])
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    titles = [{"id": convo["id"], "title": convo["title"]} for convo in data]
    return jsonify(titles)


@app.route("/history", methods=["GET"])
def get_conversation_by_id():
    user_id = request.args.get("user_id")
    convo_id = request.args.get("chat_id")
    filename = os.path.join(CONVO_DIR, f"{user_id}.json")
    if not os.path.exists(filename):
        return jsonify({"error": "Not found"}), 404
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    for convo in data:
        if convo["id"] == convo_id:
            return jsonify(convo["messages"])
    return jsonify({"error": "Chat not found"}), 404
