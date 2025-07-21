# rag_system.py
from typing import Literal, List, Dict, Any, Optional, Annotated
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
import uuid
import os
from datetime import datetime
import langdetect
from langdetect import detect, detect_langs
    
from config import Config
#from question_logger import QuestionLogger
from components.local_retriever import LocalRetriever
from components.pubMed_searcher import PubMedSearcher
from components.evaluator import Evaluator
from components.generator import Generator
from components.integrator import Integrator
from components.output_formatter import OutputFormatter
from components.memory_manager import MemoryManager
from components.bedrock_retriever import BedrockRetriever
from components.medgemma_searcher import MedGemmaSearcher

from components.parallel_searcher import ParallelSearcher


def use_last_value(current_val, new_val):
    """마지막 값만 유지하는 리듀서 함수"""
    return new_val

def append_messages(current_val: List[Dict[str, Any]], new_val: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """대화 메시지를 누적하는 리듀서 함수"""
    if current_val is None:
        current_val = []
    if new_val is None:
        return current_val
    
    result = current_val.copy()
    if isinstance(new_val, list):
        result.extend(new_val)
    return result

class GraphState(BaseModel):
    """RAG 시스템 상태를 정의하는 GraphState 클래스"""
    question: Annotated[str, use_last_value]
    documents: List[Document] = []
    generation: Optional[str] = None
    rewrite_count: int = 0
    user_id: str = "default_user"

    conversation_history: Annotated[List[Dict[str, Any]], append_messages] = []
    generation_decision: Optional[str] = None
    hallucination_decision: Optional[str] = None
    
    source_categorized_docs: Dict[str, List[Document]] = {}
    integrated_answer: Optional[str] = None
    final_formatted_output: Dict[str, Any] = {}

    original_question: Optional[str] = None

    input_language: str = ""              # 감지된 입력 언어
    target_language: str = "korean"       # 목표 출력 언어
    language_confidence: float = 0.0      # 언어 감지 신뢰도
    language_context: Dict[str, Any] = {} # 추가 언어 메타데이터

class RAGSystem:
    """리팩토링된 의료 RAG 시스템"""
    
    def __init__(self):
        self.config = Config()
        self.llm = ChatOpenAI(model=self.config.MODEL_NAME, temperature=self.config.TEMPERATURE)

        # 기본 컴포넌트 초기화
        self.evaluator = Evaluator(self.llm)
        self.generator = Generator(self.llm)
        self.integrator = Integrator(self.llm)
        self.output_formatter = OutputFormatter(self.llm)
        self.memory_manager = MemoryManager(self.llm)

        # PubMed 검색기 초기화
        self.pubmed_searcher = None
        if self.config.SEARCH_SOURCES_CONFIG.get("pubmed", False):
            self.pubmed_searcher = PubMedSearcher()
            print("✅ PubMed 검색기 초기화 완료")

        # 로컬 검색기 초기화
        self.local_retriever = None
        if self.config.SEARCH_SOURCES_CONFIG.get("local", False):
            self.local_retriever = LocalRetriever(pubmed_searcher=self.pubmed_searcher)
            self.local_retriever.set_local_search_enabled(True)
            print("✅ 로컬 검색기 초기화 완료")

        
        # S3 검색기 초기화
        self.s3_retriever = None
        if self.config.SEARCH_SOURCES_CONFIG.get("s3", False):
            from components.s3_retriever import S3Retriever
            self.s3_retriever = S3Retriever(
            bucket_name="aws-medical-chatbot",
            search_function="medical-embedding-search",
            region_name="us-east-2" 
            )
            print("✅ S3 검색기 초기화 완료")
        
        # MedGemma 초기화
        self.medgemma_searcher = None
        if self.config.SEARCH_SOURCES_CONFIG.get("medgemma", False):
            self.medgemma_searcher = MedGemmaSearcher()
            print("✅ MedGemma 검색기 초기화 완료")
        
        # Tavily 초기화
        self.tavily_searcher = None
        if self.config.SEARCH_SOURCES_CONFIG.get("tavily", False):
            from components.tavily_searcher import TavilySearcher
            self.tavily_searcher = TavilySearcher()
            print("✅ Tavily 웹 검색 초기화 완료")

        # Bedrock Retriever 추가
        bedrock_kb_id = None
        bedrock_retriever = None
        try:
            if hasattr(self.config, 'BEDROCK_CONFIG'):
                bedrock_kb_id = self.config.BEDROCK_CONFIG.get("kb_id", "")
                if bedrock_kb_id:
                    print(f"📝 Bedrock KB ID 확인됨: {bedrock_kb_id}")
                    try:
                        from components.bedrock_retriever import BedrockRetriever
                        bedrock_retriever = BedrockRetriever(
                            kb_id=bedrock_kb_id,
                            region=self.config.BEDROCK_CONFIG.get("region", "us-east-1")
                        )
                        print("✅ Bedrock Retriever 초기화 성공")
                    except Exception as e:
                        import traceback
                        print(f"⚠️ Bedrock Retriever 초기화 실패: {str(e)}")
                        print(traceback.format_exc())
                        bedrock_retriever = None
                else:
                    print("ℹ️ Bedrock KB ID가 설정되지 않았습니다")
        except Exception as e:
            print(f"⚠️ Bedrock 설정 확인 실패: {str(e)}")

        self.bedrock_retriever = bedrock_retriever

        # 병렬 검색기 초기화
        self.parallel_searcher = ParallelSearcher(
        local_retriever=self.local_retriever,
        s3_retriever=self.s3_retriever,
        medgemma_searcher=self.medgemma_searcher,
        pubmed_searcher=self.pubmed_searcher,
        tavily_searcher=self.tavily_searcher,
        bedrock_retriever=self.bedrock_retriever
    )
        
        # 워크플로우 설정
        self.workflow = None
        self.app = None
        self.checkpointer = MemorySaver()
        self._build_workflow()
    
    def _build_workflow(self):
        """간소화된 워크플로우 그래프 구성"""
        self.workflow = StateGraph(GraphState)
        
        # 핵심 노드만 설정 
        self.workflow.add_node("process_question", self._process_question)
        self.workflow.add_node("parallel_search", self._parallel_search)
        self.workflow.add_node("integrate_answers", self._integrate_answers)
        self.workflow.add_node("hallucination_check", self._hallucination_check)
        self.workflow.add_node("rewrite_question", self._rewrite_question)
        self.workflow.add_node("format_output", self._format_output)

        # 엣지 설정
        self.workflow.set_entry_point("process_question")
        self.workflow.add_edge("process_question", "parallel_search")
        self.workflow.add_edge("parallel_search", "integrate_answers")
        self.workflow.add_edge("integrate_answers", "hallucination_check")

        self.workflow.add_conditional_edges(
            "hallucination_check",
            self._get_hallucination_decision,
            {"hallucination": "rewrite_question", "relevant": "format_output"} 
        )
        
        self.workflow.add_edge("format_output", END)
        
        # 그래프 컴파일
        self.app = self.workflow.compile(checkpointer=self.checkpointer)
    
    # 노드 함수들
    def _process_question(self, state: GraphState) -> Dict[str, Any]:
        """사용자 질문 처리 및 맥락 기반 질문 재생성"""
        print("==== [PROCESS QUESTION] ====")
        
        original_question = state.question
        current_history = state.conversation_history or []

        lang_info = self._detect_language_with_confidence(original_question)
        print(f"  🌐 감지된 언어: {lang_info['language']} (신뢰도: {lang_info['confidence']:.2f})")

        # 메모리 관리
        managed_history = self.memory_manager.manage_conversation_memory(current_history)
        
        # 맥락 기반 질문 재생성
        enhanced_question = self.memory_manager.enhance_question_with_context(
            managed_history, original_question
        )
        
        # 대화 이력에 원래 질문 추가
        managed_history.append({
            "role": "user",
            "content": original_question,
            "timestamp": datetime.now().isoformat(),
            "enhanced_question": enhanced_question if enhanced_question != original_question else None,
            "language": lang_info['language'],
            "language_confidence": lang_info['confidence']
        })
        
        return {
            "conversation_history": managed_history,
            "question": enhanced_question,  # 재생성된 질문으로 검색/답변
            "original_question": original_question,
            "input_language": lang_info['language'],  
            "language_confidence": lang_info['confidence'],
            "target_language": state.target_language or "korean"  # 기본값 설정
        }
    
        
    def _parallel_search(self, state: GraphState) -> Dict[str, Any]:
        
        if self.parallel_searcher:
            categorized_docs = self.parallel_searcher.search_all_parallel(state.question)
        else:
            # 폴백: 기본 RAG 검색만
            print("  ⚠️ 병렬 검색기 없음 - RAG만 사용")
            categorized_docs = {"rag": state.documents}
        
        print(f"소스별 문서 수: {[(k, len(v)) for k, v in categorized_docs.items()]}")
        return {"source_categorized_docs": categorized_docs}
    
    def _integrate_answers(self, state: GraphState) -> Dict[str, Any]:
        """가중치 적용 답변 통합"""
        print("==== [INTEGRATE ANSWERS] ====")
        
        # 언어 정보 확인
        input_lang = state.input_language
        target_lang = state.target_language
        print(f"  🌐 입력 언어: {input_lang}, 목표 언어: {target_lang}")
        
        # 언어 정보를 integrator에 전달
        language_instruction = f"\nIMPORTANT: Respond in {target_lang}. The user asked in {input_lang}.\n"
        
        # 기존 통합 로직에 언어 지시 추가
        integrated_answer = self.integrator.integrate_answers(
            state.question + language_instruction, 
            state.source_categorized_docs
        )
        
        # 대화 이력 업데이트
        history = state.conversation_history.copy() if state.conversation_history else []
        history.append({
            "role": "assistant",
            "content": integrated_answer,
            "response_language": target_lang,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "integrated_answer": integrated_answer,
            "generation": integrated_answer,
            "conversation_history": history
        }
    
    def _hallucination_check(self, state: GraphState) -> Dict[str, Any]:
        """환각 검출 (최대 2회 재시도)"""
        print("==== [HALLUCINATION CHECK] ====")
        
        # 재시도 횟수 확인
        if state.rewrite_count >= 2:
            print("최대 재시도 횟수 도달 - 현재 답변 사용")
            return {"hallucination_decision": "relevant"}
        
        # 모든 문서 수집
        all_docs = []
        for docs in state.source_categorized_docs.values():
            all_docs.extend(docs)
        
        decision = self.evaluator.check_hallucination(
            all_docs, state.generation, state.question
        )
        
        # 환각 감지시 재시도 카운트 증가
        if decision == "hallucination":
            print(f"환각 감지 - 재시도 {state.rewrite_count + 1}/2")
            return {
                "hallucination_decision": decision,
                "rewrite_count": state.rewrite_count + 1
            }
        
        return {"hallucination_decision": decision}
    
    def _get_hallucination_decision(self, state: GraphState) -> str:
        """환각 결정 반환"""
        return state.hallucination_decision

    def _rewrite_question(self, state: GraphState) -> Dict[str, Any]:
        """질문 재작성 및 재시도"""
        print("==== [REWRITE QUESTION] ====")
        
        # 원래 질문 가져오기
        original_question = state.original_question or state.question
        
        # 질문 재작성
        rewritten_question = self.generator.rewrite_question(original_question)
        
        print(f"  원래 질문: {original_question}")
        print(f"  재작성된 질문: {rewritten_question}")
        
        # 재작성된 질문으로 업데이트
        return {"question": rewritten_question}
    
    def _format_output(self, state: GraphState) -> Dict[str, Any]:
        """최종 출력 포맷팅"""
        print("==== [FORMAT OUTPUT] ====")
        
        # 긴급도 감지 (질문과 답변에서)
        urgency_level = self._detect_urgency(state.question, state.integrated_answer)
        
        # 디바이스 타입 (추후 확장 가능)
        device_type = "desktop"  # 기본값, 추후 요청에서 가져올 수 있음
        
        # 임상 세팅
        clinical_setting = "outpatient"  # 기본값
        
        formatted_output = self.output_formatter.format_medical_answer(
            question=state.question,
            answer=state.integrated_answer,
            source_categorized_docs=state.source_categorized_docs,
            conversation_history=state.conversation_history,
            hallucination_attempts=state.rewrite_count + 1,
            original_question=state.original_question,
            urgency_level=urgency_level,
            device_type=device_type,
            clinical_setting=clinical_setting
        )
        
        return {"final_formatted_output": formatted_output}

    def _detect_urgency(self, question: str, answer: str) -> str:
        """질문과 답변에서 긴급도 자동 감지"""
        
        emergency_patterns = [
            "응급", "즉시", "emergency", "stat", "urgent", 
            "생명", "위급", "critical", "immediately"
        ]
        
        preventive_patterns = [
            "예방", "검진", "screening", "prevention", "vaccine"
        ]
        
        text = f"{question} {answer}".lower()
        
        for pattern in emergency_patterns:
            if pattern in text:
                return "emergency"
        
        for pattern in preventive_patterns:
            if pattern in text:
                return "preventive"
        
        # 추가 로직으로 urgent 감지
        if any(word in text for word in ["빠른", "신속", "곧", "soon"]):
            return "urgent"
        
        return "routine"
        
    def run_graph(self, question: str, user_id: str = None, target_language: str = "korean") -> Dict[str, Any]:
        """그래프 실행 (기존 인터페이스 유지)"""
        if not user_id:
            user_id = str(uuid.uuid4())
        
        initial_state = GraphState(
            question=question, 
            user_id=user_id,
            target_language=target_language  # 사용자가 원하는 출력 언어
        )
    
        config = {
            "configurable": {"thread_id": user_id},
            "recursion_limit": self.config.RECURSION_LIMIT
        }
        
        # 기존 대화 이력 로드
        existing_history = []
        try:
            checkpoint_tuple = self.checkpointer.get_tuple(config)
            if checkpoint_tuple:
                checkpoint = checkpoint_tuple.checkpoint
                if checkpoint and "channel_values" in checkpoint:
                    channel_values = checkpoint["channel_values"]
                    if "conversation_history" in channel_values:
                        existing_history = channel_values["conversation_history"] or []
        except Exception as e:
            print(f"기존 상태 로드 실패: {str(e)}")
        
        # 초기 상태에 기존 대화 포함
        initial_state = GraphState(
            question=question,
            user_id=user_id,
            conversation_history=existing_history,
            target_language=target_language
        )
        
        # 그래프 실행
        result = self.app.invoke(initial_state, config=config)
        
        # 결과 반환
        if "final_formatted_output" in result and result["final_formatted_output"]:
            formatted_output = result["final_formatted_output"]
            display_answer = self.output_formatter.format_for_display(formatted_output)

            # 질문 로깅
            #self._log_question_data(
            #    question, 
            #    user_id, 
            #    config.get('configurable', {}).get('thread_id', ''),
            #    display_answer,
            #    result
            #)
            
            return {
                "answer": display_answer,
                "raw_answer": result.get("generation", "답변을 생성할 수 없습니다."),
                "formatted_output": formatted_output,
                "user_id": user_id,
                "conversation_history": result.get("conversation_history", []),
                "source_breakdown": result.get("source_categorized_docs", {}),
                # 언어 정보 추가
                "language_info": {
                    "input_language": result.get("input_language", "unknown"),
                    "target_language": result.get("target_language", "korean"),
                    "confidence": result.get("language_confidence", 0.0),
                    "consistency": result.get("language_context", {}).get("consistency_score", 0.0)
                }
            }
        else:
            # 질문 로깅 (답변 실패 케이스)
            #self._log_question_data(
            #    question, 
            #    user_id, 
            #    config.get('configurable', {}).get('thread_id', ''),
            #    display_answer,
            #    result
            #)
            return {
                "answer": result["generation"] if "generation" in result else "답변을 생성할 수 없습니다.",
                "user_id": user_id,
                "conversation_history": result.get("conversation_history", []),
                "language_info": {
                    "input_language": result.get("input_language", "unknown"),
                    "target_language": result.get("target_language", "korean")
                }
            }

    def refresh_components(self):
        """
        컴포넌트를 새로고침하여 업데이트된 프롬프트 적용
        """
        print("==== [REFRESHING COMPONENTS] ====")
        
        try:
            # 주요 컴포넌트 재초기화
            self.evaluator = Evaluator(self.llm)
            self.generator = Generator(self.llm)
            self.integrator = Integrator(self.llm)
            self.output_formatter = OutputFormatter(self.llm)
            self.memory_manager = MemoryManager(self.llm)
            
            # MedGemma 검색기 재초기화 
            if self.config.SEARCH_SOURCES_CONFIG.get("medgemma", False) and hasattr(self, 'medgemma_searcher'):
                self.medgemma_searcher = MedGemmaSearcher()
            
            print("  ✅ 컴포넌트 새로고침 완료")
            return True
        except Exception as e:
            print(f"  ❌ 컴포넌트 새로고침 실패: {str(e)}")
            return False
        
    def get_stats(self) -> Dict[str, Any]:
        """시스템 통계"""
        retriever_stats = self.local_retriever.get_stats()
        
        return {
            "workflow_nodes": 6,  # 간소화된 노드 수
            **retriever_stats
        }
    
    def configure_search_sources(self, sources_config: Dict[str, bool]) -> Dict[str, bool]:
        """검색 소스 설정 업데이트"""
        print("==== [CONFIGURE SEARCH SOURCES] ====")
        
        # 병렬 검색기 소스 설정
        for source, enabled in sources_config.items():
            self.parallel_searcher.set_source_enabled(source, enabled)
        
        # 현재 설정 반환
        return self.parallel_searcher.sources_enabled

    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 및 통계 조회"""
        status = {
            "search_sources": self.parallel_searcher.sources_enabled,
            "s3_stats": self.s3_retriever.get_stats() if self.s3_retriever else None,
            "medgemma_stats": self.medgemma_searcher.get_stats() if self.medgemma_searcher else None,
        }
        
        return status
    
    def _detect_language_with_confidence(self, text: str) -> Dict[str, Any]:
        """텍스트의 언어를 감지하고 신뢰도와 함께 반환"""
        try:
            # 기본 언어 감지
            detected_lang = detect(text)
            
            # 신뢰도 포함 감지
            lang_probs = detect_langs(text)
            confidence = lang_probs[0].prob if lang_probs else 0.0
            
            # 언어 코드를 읽기 쉬운 이름으로 변환
            language_names = {
                'ko': 'korean',
                'en': 'english',
                'ja': 'japanese',
                'zh-cn': 'chinese',
                'vi': 'vietnamese',
                'th': 'thai'
            }
            
            language_name = language_names.get(detected_lang, detected_lang)
            
            return {
                'language': language_name,
                'language_code': detected_lang,
                'confidence': confidence,
                'all_probabilities': [(lang.lang, lang.prob) for lang in lang_probs]
            }
        
        except Exception as e:
            print(f"언어 감지 오류: {str(e)}")
            # 오류 시 기본값 반환
            return {
                'language': 'korean',  # 기본값
                'language_code': 'ko',
                'confidence': 0.0,
                'error': str(e)
            }

    def _check_language_consistency(self, state: GraphState) -> Dict[str, Any]:
        """각 단계에서 언어 일관성 체크 (디버깅용)"""
        checks = {
            "input_language": state.input_language,
            "target_language": state.target_language,
            "question_language": self._detect_language_with_confidence(state.question)['language'],
        }
        
        # 생성된 답변이 있으면 언어 체크
        if state.generation:
            checks["generation_language"] = self._detect_language_with_confidence(state.generation)['language']
        
        if state.integrated_answer:
            checks["integrated_answer_language"] = self._detect_language_with_confidence(state.integrated_answer)['language']
    
        # 일관성 점수 계산
        consistency_score = 1.0
        if checks.get("generation_language") and checks["generation_language"] != state.target_language:
            consistency_score -= 0.5
        
        print(f"  🔍 언어 일관성 체크: {checks}")
        print(f"  📊 일관성 점수: {consistency_score}")
        
        return {
            "language_context": {
                **state.language_context,
                "consistency_checks": checks,
                "consistency_score": consistency_score
            }
        }

    def _log_question_data(self, question: str, user_id: str, session_id: str, answer: str, result: Dict[str, Any]) -> None:
        """질문 데이터 로깅"""
        try:
            # 메타데이터 구성
            metadata = {
                "rewrite_count": result.get("rewrite_count", 0),
                "hallucination_decision": result.get("hallucination_decision", ""),
                "routing_decision": result.get("routing_decision", ""),
                "timestamp": datetime.now().isoformat()
            }
            
            # 사용된 소스 목록
            sources_used = list(result.get("source_categorized_docs", {}).keys())
            
            # 로깅 실행
            #self.question_logger.log_question(
            #    question=question,
            #    user_id=user_id,
            #    session_id=session_id,
            #    answer=answer,
            #    sources_used=sources_used,
            #    metadata=metadata
            #)
            
        except Exception as e:
            print(f"⚠️ 질문 로깅 중 오류 발생: {str(e)}")

    def refresh_components(self):
        """프롬프트 변경 후 컴포넌트 재초기화"""
        print("🔄 프롬프트 변경 감지 - 컴포넌트 재초기화 중...")
        
        # 프롬프트 의존 컴포넌트 재초기화
        from prompts import SystemPrompts
        system_prompts = SystemPrompts()
        
        # Generator 재초기화
        self.generator = Generator(self.llm)
        
        # Evaluator 재초기화
        self.evaluator = Evaluator(self.llm)
        
        # Integrator 재초기화
        self.integrator = Integrator(self.llm)
        
        # MedGemma 검색기 재초기화 (사용 중이라면)
        if hasattr(self, 'medgemma_searcher'):
            self.medgemma_searcher = MedGemmaSearcher()
        
        # Memory Manager 재초기화
        if hasattr(self, 'memory_manager'):
            self.memory_manager = MemoryManager(self.llm)
        
        print("✅ 컴포넌트 재초기화 완료")
        return True
