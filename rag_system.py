# rag_system.py (리팩토링된 버전)
from typing import Literal, List, Dict, Any, Optional, Annotated
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
import uuid
from datetime import datetime
    
from config import Config
from components.router import Router
from components.retriever import Retriever
from components.evaluator import Evaluator
from components.generator import Generator
from components.integrator import Integrator
from components.output_formatter import OutputFormatter
from components.memory_manager import MemoryManager
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
    routing_decision: Optional[str] = None
    generation_decision: Optional[str] = None
    hallucination_decision: Optional[str] = None
    
    source_categorized_docs: Dict[str, List[Document]] = {}
    integrated_answer: Optional[str] = None
    final_formatted_output: Dict[str, Any] = {}

    original_question: Optional[str] = None

class RAGSystem:
    """리팩토링된 의료 RAG 시스템"""
    
    def __init__(self):
        self.config = Config()
        self.llm = ChatOpenAI(model=self.config.MODEL_NAME, temperature=self.config.TEMPERATURE)
        
        # 핵심 컴포넌트 초기화
        self.router = Router(self.llm)
        self.retriever = Retriever()
        self.evaluator = Evaluator(self.llm)
        self.generator = Generator(self.llm)
        self.integrator = Integrator(self.llm)
        self.output_formatter = OutputFormatter()
        self.memory_manager = MemoryManager(self.llm)

        from components.medgemma_searcher import MedGemmaSearcher
        self.medgemma_searcher = MedGemmaSearcher()
        self.parallel_searcher = ParallelSearcher(self.retriever, self.medgemma_searcher)


        # 워크플로우 설정
        self.workflow = None
        self.app = None
        self.checkpointer = MemorySaver()
        self._build_workflow()
    
    def _build_workflow(self):
        """간소화된 워크플로우 그래프 구성"""
        self.workflow = StateGraph(GraphState)
        
        # 핵심 노드만 설정 (7개)
        self.workflow.add_node("process_question", self._process_question)
        self.workflow.add_node("route_question", self._route_question)
        self.workflow.add_node("retrieve", self._retrieve)
        self.workflow.add_node("parallel_search", self._parallel_search)
        self.workflow.add_node("integrate_answers", self._integrate_answers)
        self.workflow.add_node("hallucination_check", self._hallucination_check)
        self.workflow.add_node("format_output", self._format_output)

        # 간소화된 엣지 설정
        self.workflow.set_entry_point("process_question")
        self.workflow.add_edge("process_question", "route_question")
        
        self.workflow.add_conditional_edges(
            "route_question",
            self._get_route,
            {"web_search": "parallel_search", "vectorstore": "retrieve"}
        )
        
        self.workflow.add_edge("retrieve", "parallel_search")
        self.workflow.add_edge("parallel_search", "integrate_answers")
        self.workflow.add_edge("integrate_answers", "hallucination_check")
        
        self.workflow.add_conditional_edges(
            "hallucination_check",
            self._get_hallucination_decision,
            {"hallucination": "integrate_answers", "relevant": "format_output"} 
        )
        
        self.workflow.add_edge("format_output", END)
        
        # 그래프 컴파일
        self.app = self.workflow.compile(checkpointer=self.checkpointer)
    
    # 노드 함수들 (간소화된 버전)
    def _process_question(self, state: GraphState) -> Dict[str, Any]:
        """사용자 질문 처리 및 맥락 기반 질문 재생성"""
        print("==== [PROCESS QUESTION] ====")
        
        original_question = state.question
        current_history = state.conversation_history or []

        # 1단계: 메모리 관리 (요약 등)
        managed_history = self.memory_manager.manage_conversation_memory(current_history)
        
        # 2단계: 맥락 기반 질문 재생성 (이전 대화가 있으면 무조건 실행)
        enhanced_question = self.memory_manager.enhance_question_with_context(
            managed_history, original_question
        )
        
        # 3단계: 대화 이력에 원래 질문 추가 (사용자가 실제로 한 질문)
        managed_history.append({
            "role": "user",
            "content": original_question,
            "timestamp": datetime.now().isoformat(),
            "enhanced_question": enhanced_question if enhanced_question != original_question else None
        })
        
        return {
            "conversation_history": managed_history,
            "question": enhanced_question,  # 재생성된 질문으로 검색/답변
            "original_question": original_question
        }

    def _route_question(self, state: GraphState) -> Dict[str, Any]:
        """질문 라우팅"""
        print("==== [ROUTE QUESTION] ====")
        routing_decision = self.router.route_question(state.question)
        print(f"라우팅 결정: {routing_decision}")
        return {"routing_decision": routing_decision}
    
    def _get_route(self, state: GraphState) -> str:
        """라우팅 결정 반환"""
        return state.routing_decision
    
    def _retrieve(self, state: GraphState) -> Dict[str, Any]:
        """벡터 검색 (유사도 임계값 적용)"""
        print("==== [VECTOR RETRIEVE] ====")
        documents = self.retriever.retrieve_documents(state.question)
        
        # 유사도 임계값 적용
        threshold = getattr(self.config, 'SIMILARITY_THRESHOLD', 0.7)
        filtered_docs = []
        
        for doc in documents:
            similarity = doc.metadata.get("similarity_score", 0.0)
            if similarity >= threshold:
                filtered_docs.append(doc)
        
        print(f"임계값({threshold}) 이상 문서: {len(filtered_docs)}개")
        return {"documents": filtered_docs}
        
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
        print("==== [INTEGRATE WITH WEIGHTS] ====")
        
        integrated_answer = self.integrator.integrate_answers(
            state.question, state.source_categorized_docs
        )
        
        # 대화 이력 업데이트
        history = state.conversation_history.copy() if state.conversation_history else []
        history.append({
            "role": "assistant",
            "content": integrated_answer,
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
    
    def _format_output(self, state: GraphState) -> Dict[str, Any]:
        """최종 출력 포맷팅"""
        print("==== [FORMAT OUTPUT] ====")
        
        formatted_output = self.output_formatter.format_medical_answer(
            question=state.question,
            answer=state.integrated_answer,
            source_categorized_docs=state.source_categorized_docs,
            conversation_history=state.conversation_history,
            hallucination_attempts=state.rewrite_count + 1,
            original_question=state.original_question 
        )
        
        return {"final_formatted_output": formatted_output}
    
    def run_graph(self, question: str, user_id: str = None) -> Dict[str, Any]:
        """그래프 실행 (기존 인터페이스 유지)"""
        if not user_id:
            user_id = str(uuid.uuid4())
        
        initial_state = GraphState(question=question, user_id=user_id)
        
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
            conversation_history=existing_history
        )
        
        # 그래프 실행
        result = self.app.invoke(initial_state, config=config)
        
        # 결과 반환
        if "final_formatted_output" in result and result["final_formatted_output"]:
            formatted_output = result["final_formatted_output"]
            display_answer = self.output_formatter.format_for_display(formatted_output)
            
            return {
                "answer": display_answer,
                "raw_answer": result.get("generation", "답변을 생성할 수 없습니다."),
                "formatted_output": formatted_output,
                "user_id": user_id,
                "conversation_history": result.get("conversation_history", []),
                "source_breakdown": result.get("source_categorized_docs", {})
            }
        else:
            return {
                "answer": result["generation"] if "generation" in result else "답변을 생성할 수 없습니다.",
                "user_id": user_id,
                "conversation_history": result.get("conversation_history", [])
            }
    
    def load_medical_documents(self, directory_path: str) -> int:
        """의료 문서 로드 (편의 메서드)"""
        return self.retriever.load_documents_from_directory(directory_path)
    
    def refresh_components(self):
        """
        컴포넌트를 새로고침하여 업데이트된 프롬프트 적용
        """
        print("==== [REFRESHING COMPONENTS] ====")
        
        try:
            # 주요 컴포넌트 재초기화
            self.router = Router(self.llm)
            self.evaluator = Evaluator(self.llm)
            self.generator = Generator(self.llm)
            self.integrator = Integrator(self.llm)
            self.memory_manager = MemoryManager(self.llm)
            
            print("  ✅ 컴포넌트 새로고침 완료")
            return True
        except Exception as e:
            print(f"  ❌ 컴포넌트 새로고침 실패: {str(e)}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """시스템 통계"""
        retriever_stats = self.retriever.get_stats()
        
        return {
            "workflow_nodes": 7,  # 간소화된 노드 수
            **retriever_stats
        }