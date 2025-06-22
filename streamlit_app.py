# streamlit_app.py
"""
의료 AI 어시스턴트 웹 인터페이스
- 외부 접속 가능한 Streamlit 서버
- RAG 시스템과 연동
- 실시간 질문/답변
- 사용자 피드백 수집
- 프롬프트 실시간 관리
"""

import streamlit as st
import time
import json
import os
import traceback
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
from PIL import Image

ENABLE_PROMPT_EDITING = False

# 페이지 설정
st.set_page_config(
    page_title="WKUH MedLink",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* 기본 탭 스타일 */
    .stTabs [data-baseweb="tab"] {
    font-size: 24px !important;
    color: #2c3e50;
    background-color: #f0f6fb;
    padding: 20px 80px;
    border-top: 2px solid transparent;
    border-bottom: none;
    border-radius: 10px 10px 10px 10px;
    transition: background-color 0.3s ease;
    }

    /* 탭 hover 시 배경색 변경 */
    .stTabs [data-baseweb="tab"]:hover {
    background-color: #d8eafd !important; /* 밝은 하늘색 */
    color: #1f4e8c !important;
    }

    /* 선택된 탭 스타일 */
    .stTabs [aria-selected="true"] {
    background-color: #7dbdf5 !important;
    color: white !important;
    font-weight: 600 !important;
    border-bottom: none;
    }

    /* 사이드바 스타일 */
    section[data-testid="stSidebar"] {
    background-color: #f0f6fb !important;
    color: #2c3e50 !important;
    padding: 20px;
    border-right: 1px solid #dbe9f5;
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
                font-size: 17px;
                margin-left: {margin_left};
                margin-right: {margin_right};
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                line-height: 1.4;
            '>
                <b style='font-weight: 600;'>{'나' if role == "user" else 'Woni'}</b><br>{text}
            </div>
        </div>
    """, unsafe_allow_html=True)

    
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
        tb = traceback.format_exc()
        with st.expander("상세 오류 정보"):
            st.code(tb)
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
        st.exception(e)
        return None

# 프롬프트 관리자 로드
@st.cache_resource
def load_prompt_manager():
    """프롬프트 관리자 로드"""
    try:
        from prompt_manager import PromptManager
        return PromptManager()
    except Exception as e:
        st.error(f"❌ 프롬프트 관리자 로드 실패: {str(e)}")
        st.exception(e)
        return None

# 세션 상태 초기화
def initialize_session_state():
    """세션 상태 초기화"""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'user_feedback' not in st.session_state:
        st.session_state.user_feedback = []
    if 'system_stats' not in st.session_state:
        st.session_state.system_stats = {}
    if 'user_id' not in st.session_state:
        st.session_state.user_id = f"user_{int(time.time())}"
    if 'current_prompt_type' not in st.session_state:
        st.session_state.current_prompt_type = "RAG_SYSTEM_PROMPT"
    if 'edited_prompts' not in st.session_state:
        st.session_state.edited_prompts = {}
    if 'prompt_edit_history' not in st.session_state:
        st.session_state.prompt_edit_history = []

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



def display_conversation_analytics():
    """대화 분석 표시"""
    if not st.session_state.conversation_history:
        st.info("📭 아직 대화 기록이 없습니다.")
        return
    
    # 응답시간 분석
    response_times = [conv['response_time'] for conv in st.session_state.conversation_history]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 응답시간 히스토그램
        fig_hist = px.histogram(
            x=response_times,
            title="📊 응답시간 분포",
            labels={'x': '응답시간 (초)', 'y': '빈도'},
            nbins=10
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # 시간대별 사용량
        timestamps = [datetime.fromisoformat(conv['timestamp']) for conv in st.session_state.conversation_history]
        hours = [ts.hour for ts in timestamps]
        
        fig_time = px.histogram(
            x=hours,
            title="🕐 시간대별 사용량",
            labels={'x': '시간 (24시간)', 'y': '질문 수'},
            nbins=24
        )
        st.plotly_chart(fig_time, use_container_width=True)
    
    # 최근 대화들
    st.subheader("💬 최근 대화 기록")
    recent_conversations = st.session_state.conversation_history[-5:]
    
    for i, conv in enumerate(reversed(recent_conversations)):
        with st.expander(f"Q{len(recent_conversations)-i}: {conv['question'][:50]}..."):
            st.write(f"**질문:** {conv['question']}")
            st.write(f"**답변:** {conv['answer'][:200]}...")
            st.write(f"**응답시간:** {conv['response_time']:.1f}초")
            st.write(f"**시간:** {conv['timestamp'][:19]}")

def display_prompt_management_tab(rag_system, prompt_manager):
    """프롬프트 관리 탭 UI"""
    st.header("✏️ 프롬프트 관리")
    
    # 설명
    st.markdown("""
    이 페이지에서는 의료 AI 시스템의 프롬프트를 직접 수정하고 관리할 수 있습니다.
    프롬프트는 AI의 동작 방식을 결정하는 핵심 요소입니다.
    """)
    
    # 프롬프트 타입 선택
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("프롬프트 선택")
        
        # Config 클래스 임포트
        from config import Config
        
        prompt_types = list(Config.get_all_system_prompts().keys())

        # 프롬프트 타입 선택 UI
        selected_prompt_type = st.selectbox(
            "프롬프트 유형:",
            prompt_types,
            index=prompt_types.index(st.session_state.current_prompt_type) if st.session_state.current_prompt_type in prompt_types else 0
        )
        
        # 선택된 프롬프트 타입 저장
        st.session_state.current_prompt_type = selected_prompt_type
        
        # 프롬프트 설명
        prompt_descriptions = {
            "RAG_SYSTEM_PROMPT": "주요 답변 생성에 사용되는 프롬프트입니다.",
            "ROUTER_SYSTEM_PROMPT": "질문을 적절한 검색 방법으로 라우팅합니다.",
            "GRADER_SYSTEM_PROMPT": "검색된 문서의 관련성을 평가합니다.",
            "HALLUCINATION_SYSTEM_PROMPT": "생성된 답변의 환각을 검출합니다.",
            "REWRITER_SYSTEM_PROMPT": "질문을 검색에 최적화된 형태로 재작성합니다."
        }
        
        st.info(prompt_descriptions.get(selected_prompt_type, "이 프롬프트에 대한 설명이 없습니다."))
        
        # 프리셋 관리
        st.subheader("프리셋 관리")
        
        # 프리셋 저장
        preset_name = st.text_input("프리셋 이름:", key="preset_name_input")
        if st.button("현재 프롬프트 저장", key="save_preset_button"):
            if preset_name:
                # 현재 수정된 프롬프트 포함하여 저장
                all_prompts = Config.get_system_prompts()
                all_prompts.update(st.session_state.edited_prompts)
                
                success = prompt_manager.save_preset(preset_name, all_prompts)
                if success:
                    st.success(f"✅ '{preset_name}' 프리셋이 저장되었습니다!")
                else:
                    st.error("❌ 프리셋 저장에 실패했습니다.")
            else:
                st.warning("⚠️ 프리셋 이름을 입력해주세요.")
        
        # 프리셋 목록 및 로드
        presets = prompt_manager.get_preset_list()
        if presets:
            st.subheader("저장된 프리셋")
            
            for preset in presets:
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"**{preset['name']}** ({preset['prompt_count']}개 프롬프트)")
                with col_b:
                    if st.button("로드", key=f"load_{preset['name']}"):
                        loaded_prompts = prompt_manager.load_preset(preset['name'])
                        if loaded_prompts:
                            # 모든 프롬프트 업데이트
                            for p_type, p_content in loaded_prompts.items():
                                Config.update_system_prompt(p_type, p_content)
                                st.session_state.edited_prompts[p_type] = p_content
                            
                            # 새로고침 필요
                            st.success("✅ 프리셋을 로드했습니다. 적용을 위해 새로고침합니다.")
                            st.rerun()
                        else:
                            st.error("❌ 프리셋 로드에 실패했습니다.")
    
    with col2:
        st.subheader("프롬프트 편집")
        
        # 현재 프롬프트 내용 가져오기
        current_content = ""
        if selected_prompt_type in st.session_state.edited_prompts:
            current_content = st.session_state.edited_prompts[selected_prompt_type]
        else:
            current_content = getattr(Config, selected_prompt_type, "")
        
        # 프롬프트 편집 UI
        edited_content = st.text_area(
            "프롬프트 내용:",
            value=current_content,
            height=400,
            key=f"prompt_editor_{selected_prompt_type}"
        )
        
        # 변경 여부 확인
        is_changed = edited_content != current_content
        
        col_x, col_y, col_z = st.columns([1, 1, 2])
        
        with col_x:
            if st.button("적용", type="primary", disabled=not is_changed):
                # 프롬프트 업데이트
                success = Config.update_system_prompt(selected_prompt_type, edited_content)
                
                if success:
                    # 세션에 변경사항 저장
                    st.session_state.edited_prompts[selected_prompt_type] = edited_content
                    
                    # 변경 이력 추가
                    st.session_state.prompt_edit_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "prompt_type": selected_prompt_type,
                        "old_content": current_content[:100] + "..." if len(current_content) > 100 else current_content,
                        "new_content": edited_content[:100] + "..." if len(edited_content) > 100 else edited_content
                    })
                    
                    st.success("✅ 프롬프트가 업데이트되었습니다!")
                    
                    # RAG 시스템 컴포넌트 새로고침
                    if hasattr(rag_system, 'refresh_components'):
                        try:
                            rag_system.refresh_components()
                            st.success("✅ RAG 시스템이 새 프롬프트로 업데이트되었습니다!")
                        except Exception as e:
                            st.error(f"❌ RAG 시스템 업데이트 실패: {str(e)}")
                else:
                    st.error("❌ 프롬프트 업데이트에 실패했습니다.")
        
        with col_y:
            if st.button("기본값으로 복원", disabled=not is_changed):
                # 원래 Config의 프롬프트로 복원
                original_content = getattr(Config, selected_prompt_type, "")
                
                # 세션 상태에서 제거
                if selected_prompt_type in st.session_state.edited_prompts:
                    del st.session_state.edited_prompts[selected_prompt_type]
                
                # Config 업데이트
                Config.update_system_prompt(selected_prompt_type, original_content)
                
                st.success("✅ 기본값으로 복원되었습니다!")
                st.rerun()
    
    # 변경 이력 표시
    if st.session_state.prompt_edit_history:
        st.subheader("프롬프트 변경 이력")
        
        history_df = pd.DataFrame(st.session_state.prompt_edit_history)
        # 날짜 형식 변환
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # 테이블 표시
        st.dataframe(
            history_df[['timestamp', 'prompt_type', 'old_content', 'new_content']],
            column_config={
                "timestamp": "변경 시간",
                "prompt_type": "프롬프트 유형",
                "old_content": "이전 내용",
                "new_content": "새 내용"
            },
            use_container_width=True
        )
        
        if st.button("이력 초기화", key="clear_history"):
            st.session_state.prompt_edit_history = []
            st.success("✅ 변경 이력이 초기화되었습니다.")
            st.rerun()

def main():
    """메인 앱"""
    initialize_session_state()
    
    st.markdown("""
        <div style="display: inline-block; font-size: 2.5rem; font-weight: bold; margin-right: 10px; line-height: 1;">
            MedLink
        </div>
        <span style="font-size: 0.7em; color: gray; vertical-align: middle;">AI chatbot service run by Wonkwang University Hospital</span>
        """, unsafe_allow_html=True)


    st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

    # RAG 시스템 로드
    rag_system = load_rag_system()
    
    # 프롬프트 관리자 로드
    prompt_manager = None
    if ENABLE_PROMPT_EDITING:   
        prompt_manager = load_prompt_manager()
    
    if not rag_system:
        st.error("❌ 시스템을 로드할 수 없습니다. 관리자에게 문의하세요.")
        st.stop()
    
    # 사이드바
    with st.sidebar:
        st.header("TOP 10 FAQs")
        st.markdown("""
        **_**
        - 당뇨병 관리 방법은?
        - 고혈압 응급처치 절차는?
        - 심정지 환자 CPR 방법은?
        """)
        
    
    # 메인 영역 - 탭 구조
    if ENABLE_PROMPT_EDITING:
        tab1, tab2, tab3, tab4 = st.tabs(["대화 시작 (Chat)", "피드백 (Feedback)", "설정 (Settings)", "프롬프트 (Prompt)"])
    else:
        tab1, tab2, tab3 = st.tabs(["대화 시작 (Chat)", "피드백 (Feedback)", "설정 (Settings)"])

    with tab1:
        # 👋 인사말
        st.markdown("""
        <div style='
            border-radius: 12px;
            padding: 20px;
            background-color: #f2f8fc;
            margin-top: 20px;
        '>안녕하세요, 원광대학교 병원 AI 챗봇 상담사 Woni 입니다. 무엇이 궁금하신가요?</div>
        """, unsafe_allow_html=True)

        # 📝 질문 입력창
        st.markdown("""
        <div style='
            margin-top: 30px;
        '>
        """, unsafe_allow_html=True)

        question = st.text_area(
            label="",
            placeholder="질문을 입력하세요. (예: 당뇨병 관리 방법은?)",
            height=100,
            key="chat_input",
            label_visibility="collapsed"
        )

        st.markdown("</div>", unsafe_allow_html=True)


        # 버튼
        _, col1, col2, _ = st.columns([7, 1, 1, 7])
        with col1:
            submit_button = st.button("질문", type="primary")
        with col2:
            clear_button = st.button("리셋")

        if clear_button:
            st.rerun()

        answer = None  # 답변 초기화
        if submit_button and question.strip():
            if len(question.strip()) < 5:
                st.warning("구체적인 질문을 입력해주세요. (5자 이상)")
            else:
                with st.spinner("Woni가 답변을 준비 중입니다..."):
                    try:
                        start_time = time.time()
                        result = rag_system.run_graph(question, st.session_state.user_id)
                        end_time = time.time()
                        response_time = end_time - start_time

                        # 결과 처리
                        if isinstance(result, dict):
                            answer = result.get("answer", str(result))
                            sources_count = len(result.get("source_breakdown", {}).get("rag", []))
                        else:
                            answer = str(result)
                            sources_count = 0

                        # 대화 저장
                        save_conversation(question, answer, response_time, sources_count)

                    except Exception as e:
                        st.error(f"❌ 답변 생성 중 오류가 발생했습니다: {str(e)}")
                        st.exception(e)
                        st.info("💡 잠시 후 다시 시도해주세요.")

        elif submit_button:
            st.warning("⚠️ 질문을 입력해주세요.")

        # 💬 최근 대화 (최신 질문 포함)
        if st.session_state.conversation_history:
            for conv in st.session_state.conversation_history[-5:]:
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

    
    with tab2:
            st.subheader("📊 사용 통계")
            
            if st.session_state.conversation_history:
                total_questions = len(st.session_state.conversation_history)
                avg_response_time = sum(conv['response_time'] for conv in st.session_state.conversation_history) / total_questions
                
                st.metric("총 질문 수", f"{total_questions}개")
                st.metric("평균 응답시간", f"{avg_response_time:.1f}초")
                
                # 피드백 통계
                if st.session_state.user_feedback:
                    feedback_counts = {}
                    for feedback in st.session_state.user_feedback:
                        rating = feedback['rating']
                        feedback_counts[rating] = feedback_counts.get(rating, 0) + 1
                    
                    st.write("**사용자 평가:**")
                    for rating, count in feedback_counts.items():
                        st.write(f"- {rating}: {count}개")
        
        # 시스템 정보
    with tab3:
        st.header("⚙️ 시스템 설정")
        
        # 시스템 새로고침 버튼 추가
        if st.button("🔄 RAG 시스템 새로고침", type="primary"):
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
                st.subheader("🔧 설정 옵션")
            
            # 응답 모드 설정
            response_mode = st.selectbox(
                "응답 모드:",
                ["상세 답변", "간단 답변", "요약 답변"],
                index=0
            )
            
            # 안전 모드
            safety_mode = st.checkbox("🛡️ 안전 모드 (응급상황 우선 알림)", value=True)
            
            # 소스 표시
            show_sources = st.checkbox("📚 참고 문서 표시", value=True)
            
        with col2:
            st.subheader("📊 사용 통계")
            
            if st.session_state.conversation_history:
                total_questions = len(st.session_state.conversation_history)
                avg_response_time = sum(conv['response_time'] for conv in st.session_state.conversation_history) / total_questions
                
                st.metric("총 질문 수", f"{total_questions}개")
                st.metric("평균 응답시간", f"{avg_response_time:.1f}초")
                
                # 피드백 통계
                if st.session_state.user_feedback:
                    feedback_counts = {}
                    for feedback in st.session_state.user_feedback:
                        rating = feedback['rating']
                        feedback_counts[rating] = feedback_counts.get(rating, 0) + 1
                    
                    st.write("**사용자 평가:**")
                    for rating, count in feedback_counts.items():
                        st.write(f"- {rating}: {count}개")
        
        # 시스템 정보
        st.markdown("---")
        st.subheader("🖥️ 시스템 정보")
        
        if rag_system:
            stats = rag_system.get_stats()
            
            system_info = {
                "임베딩 모델": stats['model_info']['embedding_model'],
                "임베딩 차원": f"{stats['model_info']['dimensions']:,}",
                "총 문서 수": f"{stats['document_stats']['total_documents']:,}개",
                "카테고리 수": f"{stats['document_stats']['index_categories']:,}개",
                "예상 비용": f"${stats['cost_estimate']['estimated_cost_usd']:.4f}"
            }
            
            for key, value in system_info.items():
                st.write(f"**{key}:** {value}")
    
    if ENABLE_PROMPT_EDITING:
        with tab4:
            # 프롬프트 관리 탭
            if prompt_manager:
                display_prompt_management_tab(rag_system, prompt_manager)
            else:
                st.error("❌ 프롬프트 관리자를 로드할 수 없습니다.")
            
    

if __name__ == "__main__":
    main()