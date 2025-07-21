import streamlit as st
import feedparser
import os
import json
import traceback

st.set_page_config(page_title="Tester", layout="wide")


@st.cache_resource
def load_rag_system():
    try:
        from rag_system import RAGSystem
        rag_system = RAGSystem()
        return rag_system
    except Exception as e:
        st.error(f"❌ RAG 시스템 로드 실패: {str(e)}")
        return None


# 📰 뉴스용 처리
if st.query_params.get("mode") == ["news"]:
    rss_url = "https://www.who.int/feeds/entity/csr/don/en/rss.xml"
    feed = feedparser.parse(rss_url)
    st.json({"items": [ {"title": entry.title} for entry in feed.entries[:5] ]})
    st.stop()

# ✅ React → 질문 API 처리
mode = st.query_params.get("mode", [None])[0]
if mode == ["ask"]:
    payload = st.query_params.get("payload", ["{}"])[0]
    print("payload", payload)
    try:
        data = json.loads(payload)
        question = data.get("question", "")
        user_id = data.get("user_id", "default_user")

        rag = load_rag_system()
        result = rag.run_graph(question, user_id)
        if isinstance(result, dict):
            st.json({"answer": result.get("answer", "No answer")})
        else:
            st.json({"answer": str(result)})
    
    except Exception as e:
        st.json({"error": f"처리 중 오류 발생 : {str(e)}"})
    st.stop()

# 🔁 기본 React 앱 iframe 렌더
st.components.v1.iframe("http://localhost:5173", height=1000, scrolling=False)