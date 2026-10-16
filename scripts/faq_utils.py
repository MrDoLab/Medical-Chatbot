import json
from datetime import datetime
from pathlib import Path
from collections import Counter

def save_conversation_to_file(conversation_entry: dict, log_file_path: str = "./logs/streamlit_conversations.json"):
    log_file = Path(log_file_path)
    log_file.parent.mkdir(exist_ok=True)

    try:
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8") as f:
                all_convs = json.load(f)
        else:
            all_convs = []

        all_convs.append(conversation_entry)

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(all_convs, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to save conversation: {e}")

def load_conversation_history(log_file="./logs/streamlit_conversations.json"):
    log_path = Path(log_file)
    if log_path.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def get_top_faq_questions(default_questions=None, update_days=10, log_file="./logs/streamlit_conversations.json"):
    """
    최근 대화 기록 기반으로 가장 많이 등장한 질문 10개 반환.
    키워드 추출 기능은 제외. 향후 교체 가능.
    """
    updated_flag = Path("./logs/faq_last_updated.txt")
    now = datetime.now()

    if updated_flag.exists():
        last = datetime.fromisoformat(updated_flag.read_text().strip())
        if (now - last).days < update_days:
            return default_questions or []

    log_path = Path(log_file)
    if not log_path.exists():
        return default_questions or []

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            all_convs = json.load(f)

        recent = [c for c in all_convs if (now - datetime.fromisoformat(c["timestamp"])).days <= update_days]
        
        questions = [c["question"] for c in recent]
        most_common = Counter(questions).most_common(7)

        updated_flag.write_text(now.isoformat())

        return [q for q, _ in most_common]

    except Exception as e:
        print(f"[FAQ ERROR] {e}")
        return default_questions or []