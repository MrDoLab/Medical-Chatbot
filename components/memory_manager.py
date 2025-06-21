# components/memory_manager.py
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from prompts import system_prompts

class MemoryManager:
    """효율적인 대화 메모리 관리자"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.max_history = 20
        self.summary_cache = None
        self._setup_summary_chain()
        print("💭 메모리 관리자 초기화 완료")
    
    def _setup_summary_chain(self):
        """대화 요약 체인 설정"""
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompts.format("MEMORY", language="한국어")),
            
            ("human", """Summarize this medical conversation:
            
            {conversation_history}
            
            Provide a concise summary:""")
        ])
        
        self.summary_chain = self.summary_prompt | self.llm
    
    def manage_conversation_memory(self, conversation_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """대화 메모리 관리 - 20개 초과시에만 요약"""
        
        if len(conversation_history) <= self.max_history:
            # 20개 이하면 그대로 반환
            return conversation_history
        
        print(f"📝 대화 이력 관리: {len(conversation_history)}개 → 요약 + 최근 10개")
        
        # 20개 초과시: 요약 생성 (1회만)
        if self.summary_cache is None:
            old_conversations = conversation_history[:-10]  # 오래된 대화들
            self.summary_cache = self._create_conversation_summary(old_conversations)
        
        # 요약 메시지 생성
        summary_message = {
            "role": "system",
            "content": f"이전 대화 요약: {self.summary_cache}",
            "timestamp": datetime.now().isoformat(),
            "message_type": "conversation_summary"
        }
        
        # 요약 + 최근 10개 대화 반환
        recent_conversations = conversation_history[-10:]
        managed_history = [summary_message] + recent_conversations
        
        print(f"  ✅ 요약 완료: 1개 요약 + {len(recent_conversations)}개 최근 대화")
        return managed_history
    
    def _create_conversation_summary(self, conversations: List[Dict[str, Any]]) -> str:
        """대화 요약 생성 (1회만 실행)"""
        print("  🔄 대화 요약 생성 중...")
        
        try:
            # 대화 이력 포맷팅
            formatted_history = self._format_conversations(conversations)
            
            # 요약 생성
            response = self.summary_chain.invoke({
                "conversation_history": formatted_history
            })
            
            summary = response.content if hasattr(response, 'content') else str(response)
            print(f"  ✅ 요약 생성 완료: {len(conversations)}개 대화 → {len(summary)}자")
            
            return summary
            
        except Exception as e:
            print(f"  ❌ 요약 생성 실패: {str(e)}")
            return self._create_fallback_summary(conversations)
    
    def _format_conversations(self, conversations: List[Dict[str, Any]]) -> str:
        """대화 이력을 요약용으로 포맷팅"""
        formatted_parts = []
        
        for i, conv in enumerate(conversations):
            role = "👤 사용자" if conv.get("role") == "user" else "🤖 AI"
            content = conv.get("content", "")[:150]  # 150자 제한
            timestamp = conv.get("timestamp", "")[:16]  # 날짜시간 단축
            
            formatted_parts.append(f"[{timestamp}] {role}: {content}")
        
        return "\n".join(formatted_parts)
    
    def _create_fallback_summary(self, conversations: List[Dict[str, Any]]) -> str:
        """요약 생성 실패 시 폴백 요약"""
        
        # 키워드 기반 간단한 요약
        medical_keywords = []
        user_questions = 0
        
        for conv in conversations:
            if conv.get("role") == "user":
                user_questions += 1
                content = conv.get("content", "").lower()
                
                # 의료 키워드 추출
                keywords = ["당뇨", "고혈압", "응급", "약물", "치료", "증상", "진단", "수술"]
                for keyword in keywords:
                    if keyword in content and keyword not in medical_keywords:
                        medical_keywords.append(keyword)
        
        summary = f"총 {user_questions}개의 의료 질문이 있었습니다."
        
        if medical_keywords:
            summary += f" 주요 주제: {', '.join(medical_keywords[:5])}"
        
        return summary
        
    def enhance_question_with_context(self, conversation_history: List[Dict[str, Any]], current_question: str) -> str:
        """이전 대화가 있으면 무조건 맥락을 포함한 질문으로 재생성"""
        
        if not conversation_history or len(conversation_history) < 2:
            return current_question  # 첫 질문이므로 그대로 반환
        
        print(f"🔗 이전 대화 맥락 분석 중... (총 {len(conversation_history)}개 대화)")
        
        # 최근 대화 내용 추출 (최대 5개 턴)
        recent_context = self._extract_recent_context(conversation_history[-10:])
        
        # LLM을 사용해 맥락이 포함된 질문 재생성
        enhanced_question = self._generate_context_aware_question(recent_context, current_question)
        
        print(f"  📝 원래 질문: {current_question}")
        print(f"  ✨ 맥락 포함 질문: {enhanced_question[:100]}...")
        
        return enhanced_question

    def _extract_recent_context(self, recent_history: List[Dict[str, Any]]) -> str:
        """최근 대화에서 맥락 정보 추출"""
        context_parts = []
        
        for i, turn in enumerate(recent_history):
            role = turn.get("role", "")
            content = turn.get("content", "")
            
            if role == "user":
                context_parts.append(f"사용자 질문 {i+1}: {content}")
            elif role == "assistant":
                # 답변은 핵심 내용만 요약 (200자 제한)
                content_summary = content[:200] + "..." if len(content) > 200 else content
                context_parts.append(f"AI 답변 {i+1}: {content_summary}")
            elif role == "system" and turn.get("message_type") == "conversation_summary":
                context_parts.append(f"이전 대화 요약: {content}")
        
        return "\n".join(context_parts[-6:])  # 최근 6개 턴만

    def _generate_context_aware_question(self, context: str, current_question: str) -> str:
        """LLM을 사용해 맥락이 포함된 질문 생성"""
        try:
            from langchain_core.prompts import ChatPromptTemplate
            
            context_prompt = ChatPromptTemplate.from_messages([
                ("system", """당신은 의료 대화에서 맥락을 이해하여 질문을 개선하는 전문가입니다.

    이전 대화 내용을 참고하여 현재 질문을 더 명확하고 구체적으로 재작성하세요.

    지침:
    1. 이전 대화에서 언급된 의료 주제, 증상, 치료법 등을 현재 질문에 연결
    2. 애매한 지시어("그것", "이것", "그 방법" 등)를 구체적인 내용으로 교체
    3. 의료적 맥락을 명확히 하여 더 정확한 답변이 가능하도록 개선
    4. 한국어로 자연스럽게 작성
    5. 원래 질문의 의도는 유지하되 맥락 정보 추가

    예시:
    - 원래: "부작용은 뭐가 있어?"
    - 개선: "앞서 언급한 메트포민의 부작용은 뭐가 있어?"

    - 원래: "다른 방법은?"  
    - 개선: "당뇨병 관리에서 약물 치료 외의 다른 방법은?"
    """),
                ("human", """이전 대화 맥락:
    {context}

    현재 질문: {current_question}

    위 맥락을 참고하여 현재 질문을 더 명확하고 구체적으로 재작성해주세요:""")
            ])
            
            enhanced_question = self.llm.invoke(
                context_prompt.format(context=context, current_question=current_question)
            ).content
            
            return enhanced_question.strip()
            
        except Exception as e:
            print(f"  ❌ 맥락 질문 생성 실패: {str(e)}")
            # 폴백: 간단한 맥락 추가
            return self._simple_context_enhancement(context, current_question)

    def _simple_context_enhancement(self, context: str, current_question: str) -> str:
        """LLM 실패 시 간단한 맥락 추가"""
        # 최근 사용자 질문 추출
        recent_topics = []
        for line in context.split('\n'):
            if '사용자 질문' in line:
                topic = line.split(': ', 1)[-1]
                recent_topics.append(topic)
        
        if recent_topics:
            latest_topic = recent_topics[-1][:50]
            return f"이전 대화에서 '{latest_topic}'에 대해 논의했는데, {current_question}"
        
        return f"이전 대화 맥락에서 이어서, {current_question}"

    def reset_summary_cache(self):
        """요약 캐시 초기화 (필요시 사용)"""
        self.summary_cache = None
        print("🔄 요약 캐시 초기화됨")