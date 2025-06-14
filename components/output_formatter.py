# components/output_formatter.py 
from typing import Dict, Any, List
from langchain_core.documents import Document
from datetime import datetime

class OutputFormatter:
    """기본적인 의료 답변 포맷터"""
    
    def __init__(self):
        """출력 포맷터 초기화"""
        print("📝 출력 포맷터 초기화 완료")
    
    def format_medical_answer(self, 
                            question: str, 
                            answer: str, 
                            source_categorized_docs: Dict[str, List[Document]],
                            conversation_history: List[Dict] = None,
                            hallucination_attempts: int = 1,
                            original_question: str = None) -> Dict[str, Any]:
        """의료 답변을 기본 포맷으로 구성"""
        print("📝 최종 답변 포맷팅")
        
        # 소스 정보 구성
        sources_info = self._build_sources_info(source_categorized_docs)
        
        # 최종 포맷된 답변 구성
        formatted_answer = self._build_formatted_answer(
            answer, sources_info, hallucination_attempts, original_question
        )
  
        return {
            "main_answer": formatted_answer,
            "sources_used": sources_info["source_list"],
            "total_sources": sources_info["total_count"],
            "metadata": {
                "question": question,
                "generated_at": datetime.now().isoformat(),
                "hallucination_checks": hallucination_attempts,
                "source_breakdown": sources_info["breakdown"]
            }
        }
    
    def _build_sources_info(self, source_categorized_docs: Dict[str, List[Document]]) -> Dict[str, Any]:
        """소스 정보 구성"""
        source_list = []
        breakdown = {}
        total_count = 0
        
        for source_type, docs in source_categorized_docs.items():
            doc_count = len(docs)
            breakdown[source_type] = doc_count
            total_count += doc_count
            
            for doc in docs:
                source_list.append({
                    "type": source_type,
                    "source": doc.metadata.get("source", "unknown"),
                    "title": doc.metadata.get("title", "제목 없음")
                })
        
        return {
            "source_list": source_list,
            "breakdown": breakdown,
            "total_count": total_count
        }
    
    def _build_formatted_answer(self, answer: str, sources_info: Dict[str, Any], 
                              hallucination_attempts: int, original_question: str = None) -> str:

        """최종 포맷된 답변 구성"""
        
        # 기본 답변
        formatted_parts = [answer.strip()]
        
        # 소스 정보 추가
        if sources_info["total_count"] > 0:
            formatted_parts.append(f"\n📚 **참고 자료**: {sources_info['total_count']}개 문서")
            
            # 소스별 개수 표시
            breakdown = sources_info["breakdown"]
            source_details = []
            
            if breakdown.get("pubmed", 0) > 0:
                source_details.append(f"PubMed 논문 {breakdown['pubmed']}개")
            if breakdown.get("rag", 0) > 0:
                source_details.append(f"내부 DB {breakdown['rag']}개")
            if breakdown.get("web", 0) > 0:
                source_details.append(f"웹 자료 {breakdown['web']}개")
            
            if source_details:
                formatted_parts.append(f"• {', '.join(source_details)}")
        
        # 품질 검증 정보
        if hallucination_attempts > 1:
            formatted_parts.append(f"\n🔍 **품질 검증**: {hallucination_attempts}회 검토 완료")
        

        # 맥락 정보 추가 (간단 버전)
        if original_question and original_question != answer:
            formatted_parts.append(f"\n🔗 **원래 질문**: {original_question}")

        # 기본 의료 면책 조항
        formatted_parts.append("\n💡 **안내**: 이 정보는 의학적 조언을 대체할 수 없습니다. 정확한 진단과 치료를 위해 의료 전문가와 상담하세요.")
        
        # 응급상황 체크
        if any(keyword in answer.lower() for keyword in ["응급", "급성", "위험", "심각", "즉시"]):
            formatted_parts.append("\n🚨 **응급상황 시**: 119 또는 가까운 응급실로 즉시 연락하세요.")
        
        return "\n".join(formatted_parts)
    
    def format_for_display(self, formatted_output: Dict[str, Any]) -> str:
        """사용자 표시용 최종 텍스트 형태로 변환"""
        return formatted_output.get("main_answer", "답변을 생성할 수 없습니다.")