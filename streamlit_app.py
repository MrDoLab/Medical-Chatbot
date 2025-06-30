# streamlit_app.py
"""
- RAG 시스템과 연동
"""

import streamlit as st
import time
import json
import traceback
from datetime import datetime
from pathlib import Path
from PIL import Image
import base64
from faq_utils import save_conversation_to_file, get_top_faq_questions, load_conversation_history
import requests
import feedparser


loading_container = st.empty()

if "initial_loading_done" not in st.session_state:
    st.session_state["initial_loading_done"] = False

if not st.session_state["initial_loading_done"]:
    with loading_container.container():
        
        st.markdown("""
            <div style='
                    font-size: 45px;
                    color: #333;
                    line-height: 0.8;
                    padding-left: 15px;
                    font-weight: 500;
                    margin-top: 300px;
                    '>
                    Welcome Back to <span style='color:#003366;'>WKUH MedLink...</span><br>
                    <small style='font-size:18px; color:#7F8C8D;'>AI Chatbot Assistant for Smarter Decisions</small>
            </div>
            """, unsafe_allow_html=True)
    time.sleep(2)
    loading_container.empty()
    st.session_state["initial_loading_done"] = True


texts = {"input_placeholder_ko": "질문을 입력하세요. (예: 당뇨병 관리 방법은?)",
         "input_placeholder_en": "Type your question here (e.g., How to manage diabetes?)"}



# 페이지 설정
st.set_page_config(
    page_title="WKUH MedLink",
    page_icon = "hlogo.png",
    layout="wide",
    initial_sidebar_state="expanded"    
)

st.markdown("""
    <style>

    /* 탭 강조 색상 변경 */
    .stTabs [aria-selected="true"] {
        background-color: #003366 !important;
        color: white !important;
    }
    button[kind="primary"] {
        background-color: #003366 !important;
        color: white !important;
        border-radius: 8px;
        font-size: 18px !important;
    }
    

    /* 기본 탭 스타일 */
    .stTabs [data-baseweb="tab"] {
        padding: 25px 70px !important;
        font-size: 38px !important;
        color: #003366;
        background-color: #f0f6fb;
        border-top: 2px solid transparent;
        border-radius: 8px 8px 0px 0px;
        border-bottom: none;
        transition: background-color 0.3s ease;
    }
    

    .stTabs [data-baseweb="tab"] div {
        font-size: 20px !important;
}
    /* 탭 hover 시 배경색 변경 */
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #d8eafd !important; 
        color: #1f4e8c !important;
    }

    /* 선택된 탭 스타일 */

    textarea {
        font-size: 20px !important;
    }
    
    /* 사이드바 스타일 */
    section[data-testid="stSidebar"] {
    width: 330px !important;
    resize: none !important;
    background-color: #f0f6fb !important;
    color: #2c3e50 !important;
    align-items: left;
    padding: 20px;
    border-right: 1px solid #dbe9f5;
    }
    
    
    /* 여기 사이드바 FAQ 버튼 텍스트 */
    section[data-testid="stSidebar"] .stButton button div{
        font-size: 18px !important;
        text-align: left !important;
        word-break: keep-all !important;
    }
    /* 버튼내부 span */
    section[data-testid="stSidebar"] .stButton button span {
        font-size: 20px !important;
    }
    /* 의료 뉴스 링크 */
    section[data-testid="stSidebar"] a {
        font-size: 18px !important;
    }


    
    section[data-testid="stSidebar"] .stMarkdown p {
    color: #2c3e50 !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
    color: #1f2937 !important;
    }
    </style>
""", unsafe_allow_html=True)




def get_medical_news(n=3):
    rss_url = "https://www.koreabiomed.com/rss/allArticle.xml"
    news_list = []

    try:
        feed = feedparser.parse(rss_url)
        for entry in feed.entries[:n]:
            title = entry.title
            link = entry.link
            news_list.append((title, link))
        return news_list

    except Exception as e:
        st.warning(f"의료 뉴스를 불러오는 데 실패했습니다: {e}")
        return []
    
def render_chat_bubble(role: str, text: str):
    align = "right" if role == "user" else "left"
    bubble_color = "#f0f6fb" if role == "user" else "#f0f2f6"
    text_color = "#333" if role == "user" else "#333"
    border_radius = "20px 20px 0 20px" if role == "user" else "20px 20px 20px 0"
    margin_left = "20%" if role == "user" else "0"
    margin_right = "0" if role == "user" else "20%"
    
    st.markdown(f"""
        <div style='text-align: {align}; margin: 10px 0;'>
            <div style='
                display: inline-block; 
                background-color: {bubble_color};
                color: {text_color};
                padding: 14px 20px;
                border-radius: {border_radius};
                max-width: 75%;
                font-size: 20px;
                margin-left: {margin_left};
                margin-right: {margin_right};
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                line-height: 1.4;
            '>
                <b style='font-weight: 600;'>{'나' if role == "user" else 'Woni'}</b><br>{text}
            </div>
        </div>
    """, unsafe_allow_html=True)

    
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
    
def initialize_session_state():
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = load_conversation_history()
    if 'user_feedback' not in st.session_state:
        st.session_state.user_feedback = []
    if 'user_id' not in st.session_state:
        st.session_state.user_id = f"user_{int(time.time())}"
    if 'intro_shown' not in st.session_state:
        st.session_state.intro_shown = False
    if 'display_history' not in st.session_state:
        st.session_state.display_history = []  # 화면에 잠깐 보여줄 대화 리스트

        
# RAG 시스템 로드 (캐시로 한 번만 로드)
@st.cache_resource
def load_rag_system():
    """RAG 시스템 로드 (한 번만 실행)"""
    try:
        from rag_system import RAGSystem
        rag_system = RAGSystem()
        return rag_system
    except Exception as e:
        st.error(f"❌ RAG 시스템 로드 실패: {str(e)}")
        return None

# QA 평가기 로드
@st.cache_resource
def load_qa_evaluator():
    """QA 평가기 로드"""
    try:
        from qa_evaluator import MedicalQAEvaluator
        return MedicalQAEvaluator()
    except Exception as e:
        st.error(f"❌ QA 평가기 로드 실패: {str(e)}")
        return None

# 세션 상태 초기화


def save_conversation(question: str, answer: str, response_time: float, sources: int = 0):
    """대화 저장"""
    conversation_entry = {
        'timestamp': datetime.now().isoformat(),
        'question': question,
        'answer': answer,
        'response_time': response_time,
        'sources_used': sources,
        'user_id': st.session_state.user_id
    }
    st.session_state.conversation_history.append(conversation_entry)
    save_conversation_to_file(conversation_entry)


def save_feedback(question: str, answer: str, rating: str, feedback_text: str = ""):
    """사용자 피드백 저장"""
    feedback_entry = {
        'timestamp': datetime.now().isoformat(),
        'question': question,
        'answer': answer,
        'rating': rating,
        'feedback_text': feedback_text,
        'user_id': st.session_state.user_id
    }
    st.session_state.user_feedback.append(feedback_entry)
    
    # 로컬 파일로도 저장
    feedback_file = Path("./logs/streamlit_feedback.json")
    feedback_file.parent.mkdir(exist_ok=True)
    
    try:
        if feedback_file.exists():
            with open(feedback_file, 'r', encoding='utf-8') as f:
                all_feedback = json.load(f)
        else:
            all_feedback = []
        
        all_feedback.append(feedback_entry)
        
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(all_feedback, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"피드백 저장 실패: {e}")


    # 최근 대화들
    st.subheader("💬 최근 대화 기록")
    recent_conversations = st.session_state.conversation_history[-5:]
    
    for i, conv in enumerate(reversed(recent_conversations)):
        with st.expander(f"Q{len(recent_conversations)-i}: {conv['question'][:50]}..."):
            st.write(f"**질문:** {conv['question']}")
            st.write(f"**답변:** {conv['answer'][:200]}...")
            st.write(f"**응답시간:** {conv['response_time']:.1f}초")
            st.write(f"**시간:** {conv['timestamp'][:19]}")   

def main():
    """메인 앱"""
    initialize_session_state()
    
    if "lang" not in st.session_state:
        st.session_state["lang"] = "ko"  # 기본 한국어

    lang = st.session_state["lang"]
    lang_placeholder = st.empty()
    
    if st.session_state.get("lang_changing", False):
        st.session_state["lang_changing"] = False
        loading_container = st.empty()
        with loading_container.container():
            st.markdown("""
                    <div style = '
                        display:flex;
                        align-items: center;
                        flex-direction: column;
                        margin-top: 90px;
                        justify-content: center;
                        font-size: 22px;
                        text-align: center;
                        color:#7F8C8D;
                    '>
                        언어 설정을 변경하는 중입니다... 잠시만 기다려 주세요.
                    </div>
                    """, unsafe_allow_html=True)
            time.sleep(1)
            st.rerun()
    
    with lang_placeholder.container():
        st.markdown("<div class='lang-buttons'>", unsafe_allow_html= True)
        _, col2, col3, col4 = st.columns([17, 1, 0.8, 1])
        
        with col3:
            if st.button("한글", key="btn_ko"):  # 한국 국기
                st.session_state["lang"] = "ko"
                st.session_state["lang_changing"] = True
                st.rerun()
                
        with col4:
            if st.button("English", key="btn_en"):  # 미국 국기
                st.session_state["lang"] = "en"
                st.session_state["lang_changing"] = True
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    
    image_base64 = get_base64_image("hlogo.png")

# 원래 pic height 78px

    st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 30px;">
            <img src="data:image/png;base64,{image_base64}" 
                style="height: 135px; border-radius: 14px;" />
            <div style="display: flex; flex-direction: column;">
                <div style="font-size: 3.1rem; color: #003366; font-weight: bold; line-height: 1;">MedLink</div>
                <span style="font-size: 1.2em; font-weight: light-bold; color: #003366; margin-top: 7px;">제생의세(濟生醫世) 정신으로 의술로써 병든 세상을 구한다</span>
                <span style="font-size: 1.1em; color: gray;">AI chatbot service run by Wonkwang University Hospital</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    


    st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
    
    # RAG 시스템 로드
    rag_system = load_rag_system()

    if not rag_system:
        st.error("❌ 시스템을 로드할 수 없습니다. 관리자에게 문의하세요.")
        st.stop()
        
    base_faq = [
        "폐렴 치료에서 CURB-65 점수의 해석은?",
        "WPW syndrome의 금기 약물은?",
        "SIADH의 진단 기준은 무엇인가?",
        "Kawasaki disease의 진단 기준과 치료는?",
        "의식저하 환자에서 hypoglycemia rule-out 순서는?",
        "Parkinson 병의 cardinal signs는?",
        "Trauma 환자에서 GCS 계산 방법은?",
    ]

    #잠시 비활성화
    faq_questions = get_top_faq_questions(default_questions=base_faq, update_days=10)

    with st.sidebar:
        st.markdown("""
            <div style='font-size: 22px; font-weight: bold;'><br>자주 묻는 질문</div>
            """  if lang=="ko" else """
            <div style='font-size: 24px; font-weight: bold;'><br>TOP 7 FAQs</div>
            """, unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size: 16px; color: gray;">* 10일 주기로 업데이트됩니다.</p>' if lang=="ko"
            else '<p style="font-size: 16px; color: gray;">* Updates every 10 days.</p>' ,
            unsafe_allow_html=True)

        for i, question in enumerate(faq_questions):
            if st.button(question, key=f"faq_{i}"):
                st.session_state.chat_input = question
                st.session_state.trigger_faq_submit = True

        # 👇 최신 뉴스 3개 세로로 추가
        news = get_medical_news(n=3)
        if news:
            st.markdown("""<hr>
            <div style='margin-top: 20px; font-size: 22px; font-weight: bold;'><br>최근 의료 소식</div>
            """  if lang=="ko" else """<hr>
            <div style='margin-top: 20px; font-size: 24px; font-weight: bold;'><br>Medical News</div>
            """, unsafe_allow_html=True)

            for title, link in news:
                short_title = title[:60] + "..." if len(title) > 60 else title
                st.markdown(f"""
                    <a href="{link}" target="_blank" 
                    style="
                        display: block;
                        background-color: white;
                        color: #2c3e50;
                        font-size: 18px;
                        padding: 10px 12px;
                        border-radius: 8px;
                        text-decoration: none;
                        margin-top: 10px;
                        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
                        border: 1px solid #90caf9;
                    ">
                        {short_title}
                    </a>
                """, unsafe_allow_html=True)
                
            
            
            st.markdown("""
                <hr>
                <div style='text-align: center; font-size: 18px; color: gray; margin-top: 30px;'>
                    실시간 의료 뉴스는 코리아바이오메드 (Korea Biomedical Review)에서 제공합니다.<br><br><br>
                    <b>WKUH MedLink v1.0</b><br>
                    최종 업데이트: 2025.06.29<br><br>
                </div>
                """, unsafe_allow_html=True)
    
    # 메인 영역 - 탭 구조 수정
    tab1, tab2, tab3 = st.tabs([
        "질문" if lang =="ko" else "Ask",
        "지난 대화" if lang=="ko" else "Chat History",
        "⚙️ 설정" if lang=="ko" else "⚙️ Settings"])
   
        
    with tab1:
        # 👋 인사말 박스 전체
        message = "안녕하세요, 원광대학교 병원 AI 챗봇 상담사 Woni 입니다. 무엇이 궁금하신가요?" if lang=="ko" else "Hello, I am Woni, AI chatbot from WKUH. How can I help you?"

        if not st.session_state.intro_shown:
            placeholder = st.empty()
            typed_text = ""

            for char in message:
                typed_text += char
                placeholder.markdown(f"""
                <div style='
                    border-left: 6px solid var(--primary-color);
                    border-radius: 8px;
                    padding: 16px;
                    background-color: var(--accent-color);
                    margin-top: 20px;
                    line-height: 1.6;
                '>
                    <div style='font-size: 20px; line-height: 1.4;'>{typed_text}</div>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(0.02)

            st.session_state.intro_shown = True  # 타이핑 효과 딱 한 번만 실행
        else:
            st.markdown(f"""
            <div style='
                border-left: 6px solid var(--primary-color);
                border-radius: 8px;
                padding: 16px;
                background-color: var(--accent-color);
                margin-top: 20px;
                line-height: 1.6;
            '>
                <div style='font-size: 20px; line-height: 1.4;'>{message}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # 📝 질문 입력창
        st.markdown("""
        <div style='
            margin-top: 30px;
        '>
        """, unsafe_allow_html=True)
            
        question = st.text_area(
            label="",
            placeholder=texts['input_placeholder_ko'] if lang == "ko" else texts['input_placeholder_en'],
            height=100,
            key="chat_input",
            label_visibility="collapsed"
        )

        st.markdown("</div>", unsafe_allow_html=True)
        
        # 버튼
        _, col1, col2, _ = st.columns([7, 1, 1, 7])
        
        if st.session_state.get("trigger_faq_submit", False):
            question = st.session_state.get("chat_input", "")
            st.session_state.trigger_faq_submit = False
            submit_button = True
        else:
            with col1:
                submit_button = st.button("질문", type="primary", help="질문 보내기")
        with col2:
            clear_button = st.button("리셋", help="새로운 대화")

        if clear_button:
            st.session_state.display_history = []
            st.session_state.intro_shown = False
            st.rerun()

        answer = None  # 답변 초기화
        if submit_button and question.strip():
            if len(question.strip()) < 5:
                st.warning("구체적인 질문을 입력해주세요. (5자 이상)")
            else:
                with st.spinner(""):
                    try:
                        start_time = time.time()
                        result = rag_system.run_graph(question, st.session_state.user_id)
                        end_time = time.time()
                        response_time = end_time - start_time

                        if isinstance(result, dict):
                            answer = result.get("answer", str(result))
                            sources_count = len(result.get("source_breakdown", {}).get("rag", []))
                        else:
                            answer = str(result)
                            sources_count = 0

                        # 전체 대화 저장
                        save_conversation(question, answer, response_time, sources_count)

                        # 화면 출력용 대화만 따로 관리
                        st.session_state.display_history.append({
                            "question": question,
                            "answer": answer
                        })

                    except Exception as e:
                        st.error(f"❌ 답변 생성 중 오류가 발생했습니다: {str(e)}")
                        st.info("잠시 후 다시 시도해주세요.")


        elif submit_button:
            st.warning("질문을 입력해주세요.")

        # 💬 최근 대화 (최신 질문 포함)
        if st.session_state.display_history:
            for conv in st.session_state.display_history:
                render_chat_bubble("user", conv['question'])
                render_chat_bubble("assistant", conv['answer'])

        # 🌟 피드백
        if answer:
            st.markdown("---")
            st.subheader("답변이 마음에 드셨나요?")
            col1, col2, col3, col4 = st.columns(4)
            feedback_given = False

            with col1:
                if st.button("😊 매우 좋음"):
                    save_feedback(question, answer, "excellent")
                    st.success("피드백 감사합니다! 😊")
                    feedback_given = True

            with col2:
                if st.button("👍 좋음"):
                    save_feedback(question, answer, "good")
                    st.success("피드백 감사합니다! 👍")
                    feedback_given = True

            with col3:
                if st.button("😐 보통"):
                    save_feedback(question, answer, "average")
                    st.info("피드백 감사합니다! 😐")
                    feedback_given = True

            with col4:
                if st.button("😞 별로"):
                    save_feedback(question, answer, "poor")
                    st.warning("피드백 감사합니다! 개선하겠습니다. 😞")
                    feedback_given = True

            if feedback_given:
                additional_feedback = st.text_area("추가 의견이 있으시면 입력해주세요:", key="additional_feedback")
                if st.button("의견 제출") and additional_feedback:
                    if st.session_state.user_feedback:
                        st.session_state.user_feedback[-1]['feedback_text'] = additional_feedback
                    st.success("추가 의견이 저장되었습니다!")

    

    with tab3:
        st.markdown("""
            <div style= '
            font-size: 24px;
            font-weight: bold;
            margin-top: 20px;
            margin-bottom: 15px;
            '>
                시스템 설정
            </div>
            """, unsafe_allow_html=True)

        # 시스템 새로고침 버튼 추가 
        if st.button("🔄 RAG 새로고침", type="primary"):
            try:
                # 모든 캐시 리소스 초기화
                st.cache_resource.clear()
                st.success("✅ RAG 시스템 캐시가 초기화되었습니다. 새로고침됩니다.")
                st.rerun()
            except Exception as e:
                st.error(f"❌ 캐시 초기화 실패: {str(e)}")
                st.info("💡 페이지를 수동으로 새로고침해보세요.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("설정 옵션")
            
            response_mode = st.selectbox(
                "응답 모드:",
                ["상세 답변", "간단 답변", "요약 답변"],
                index=0
            )
            
            safety_mode = st.checkbox("🛡️ 안전 모드 (응급상황 우선 알림)", value=True)
            show_sources = st.checkbox("📚 참고 문서 표시", value=True)
            
        
        
    with tab2:
        if lang == "ko":
            text = """
            <div id="history-area" style='
                font-size: 24px;
                font-weight: bold;
                margin-top: 20px;
                margin-bottom: 15px;
            '>
                대화 기록 검색
            </div>
            """
        else:
            text = """
            <div id="history-area" style='
                font-size: 24px;
                font-weight: bold;
                margin-top: 20px;
                margin-bottom: 15px;
            '>
                Search Chat History
            </div>
            """

        st.markdown(text, unsafe_allow_html=True)


        if st.session_state.conversation_history:
            total_questions = len(st.session_state.conversation_history)

            st.markdown(f"""
                <div style='
                    font-size: 20px;
                    margin-bottom: 20px;
                '>
                    총 <span style="font-weight: bold;">{total_questions}</span>개의 대화가 저장되어 있습니다.
                </div>
            """, unsafe_allow_html=True)

            # 검색창
            st.markdown("""
                <div style='
                    font-size: 18px;
                    margin-bottom: 5px;
                '>검색 키워드를 입력하세요</div>
            """, unsafe_allow_html=True)

            keyword_input = st.text_input(
                label="",  
                placeholder="키워드 입력 후 엔터",
                label_visibility="collapsed"
            )

            st.markdown("""
                <div style='
                    font-size: 18px;
                    margin-top: 10px;
                    margin-bottom: 5px;
                '>검색일 수 (일)</div>
            """, unsafe_allow_html=True)

            days_filter = st.slider("", min_value=1, max_value=90, value=30, step=1, label_visibility="collapsed")

            now = datetime.now()
            
            # 필터링된 대화 리스트
            filtered = []
            for conv in reversed(st.session_state.conversation_history):
                timestamp = datetime.fromisoformat(conv["timestamp"])
                if (now - timestamp).days > days_filter:
                    continue

                if keyword_input:
                    keywords = [k.strip().lower() for k in keyword_input.split()]
                    q_lower = conv["question"].lower()
                    a_lower = conv["answer"].lower()
                    
                    if all(kw in q_lower or kw in a_lower for kw in keywords):
                        filtered.append(conv)
                else:
                    filtered.append(conv)

            st.markdown(f"<div style='font-size: 20px; margin-top: 60px; margin-bottom: 10px;'><b>기간 내 검색결과 : {len(filtered)}건</b><br>", unsafe_allow_html=True)
            
            # 스택형 아코디언 출력
            for idx, conv in enumerate(filtered, start=1):
                st.markdown("""
                <style>
                div [role="button"] > div {
                    font-size:22px;
                    padding: 25px;
                    line-height: 2;
                }
                </style>
                """, unsafe_allow_html= True)
                with st.expander(f"Q{idx}: {conv['question'][:50]}..."):
                    st.markdown(f"""
                        <div style='
                            background-color: #f9fcff;
                            padding: 25px;
                            border-radius: 8px;
                            border: 1px solid #dbe9f5;
                            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
                            margin-bottom: 10px;
                            line-height: 1.6;
                            font-size: 20px;
                        '>
                            <b style='color: #2c3e50;'>질문:</b> {conv['question']}<br><br>
                            <b style='color: #2c3e50;'>답변:</b> {conv['answer']}<br><br>
                            <b style='color: #2c3e50;'>시간:</b> {conv['timestamp'][:19]}<br>
                        </div>
                    """, unsafe_allow_html=True)

            
        else:
            st.info("대화 기록이 없습니다.")

   

            
if __name__ == "__main__":
    main()