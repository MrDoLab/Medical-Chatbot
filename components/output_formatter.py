# components/output_formatter.py (개선 버전)
from typing import Dict, Any, List
from langchain_core.documents import Document
from datetime import datetime
import re

class OutputFormatter:
    """의료 전문가용 상세 답변 포맷터"""
    
    def __init__(self):
        """출력 포맷터 초기화"""
        print("📝 의료 전문가용 출력 포맷터 초기화 완료")
        
        # 언어 감지 패턴
        self.korean_pattern = re.compile('[가-힣]')
        self.english_pattern = re.compile('[a-zA-Z]')
        
    def format_medical_answer(self, 
                            question: str, 
                            answer: str, 
                            source_categorized_docs: Dict[str, List[Document]],
                            conversation_history: List[Dict] = None,
                            hallucination_attempts: int = 1,
                            original_question: str = None) -> Dict[str, Any]:
        """의료 답변을 전문가용 포맷으로 구성"""
        print("📝 의학 전문가용 답변 포맷팅")
        
        # 소스 정보 구성
        sources_info = self._build_sources_info(source_categorized_docs)
        
        # 참고문헌 목록 생성
        references = self._build_references_list(sources_info["source_list"])
        
        # 언어 감지
        target_language = self._detect_language(question)
        
        # 최종 포맷된 답변 구성
        formatted_answer = self._build_formatted_answer(
            answer, sources_info, references, hallucination_attempts, target_language, original_question
        )
  
        return {
            "main_answer": formatted_answer,
            "sources_used": sources_info["source_list"],
            "total_sources": sources_info["total_count"],
            "references": references,
            "target_language": target_language,
            "metadata": {
                "question": question,
                "generated_at": datetime.now().isoformat(),
                "hallucination_checks": hallucination_attempts,
                "source_breakdown": sources_info["breakdown"]
            }
        }
    
    def _detect_language(self, text: str) -> str:
        """텍스트 언어 감지"""
        if not text:
            return "english"  # 기본값
            
        korean_count = len(self.korean_pattern.findall(text))
        english_count = len(self.english_pattern.findall(text))
        
        if korean_count > english_count:
            return "korean"
        else:
            return "english"
    
    def _build_sources_info(self, source_categorized_docs: Dict[str, List[Document]]) -> Dict[str, Any]:
        """소스 정보 구성"""
        source_list = []
        breakdown = {}
        total_count = 0
        
        for source_type, docs in source_categorized_docs.items():
            doc_count = len(docs)
            breakdown[source_type] = doc_count
            total_count += doc_count
            
            for i, doc in enumerate(docs):
                source_info = {
                    "id": len(source_list) + 1,  # 순차적 ID 부여
                    "type": source_type,
                    "source": doc.metadata.get("source", "unknown"),
                    "title": doc.metadata.get("title", "제목 없음"),
                    "authors": doc.metadata.get("authors", ""),
                    "year": doc.metadata.get("year", ""),
                    "journal": doc.metadata.get("journal", ""),
                    "url": doc.metadata.get("url", ""),
                    "doi": doc.metadata.get("doi", ""),
                    "similarity_score": doc.metadata.get("similarity_score", 0),
                    "content_preview": doc.page_content[:150] if hasattr(doc, 'page_content') else ""
                }
                source_list.append(source_info)
        
        # 유사도 점수로 소스 정렬
        source_list.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # ID 재부여
        for i, source in enumerate(source_list):
            source["id"] = i + 1
        
        return {
            "source_list": source_list,
            "breakdown": breakdown,
            "total_count": total_count
        }
    
    def _build_references_list(self, source_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """참고문헌 목록 생성"""
        references = []
        
        for source in source_list:
            ref_id = source["id"]
            source_type = source["type"]
            
            # 소스 타입별 참고문헌 포맷 다르게 처리
            if source_type == "pubmed":
                # 학술 논문 형식
                authors = source.get("authors", "")
                if isinstance(authors, list):
                    authors = ", ".join(authors[:3])
                    if len(source.get("authors", [])) > 3:
                        authors += " et al."
                        
                reference = {
                    "id": ref_id,
                    "text": f"[PubMed] {authors}. {source.get('title', '제목 없음')}. {source.get('journal', '')}. {source.get('year', '')}. DOI: {source.get('doi', 'N/A')}"
                }
                
            elif source_type == "bedrock_kb":
                # Knowledge Base 형식
                reference = {
                    "id": ref_id,
                    "text": f"[Bedrock KB] {source.get('title', '제목 없음')}. Document ID: {source.get('source', 'unknown')}. {source.get('year', '')}"
                }
                
            elif source_type == "tavily":
                # 웹 검색 결과 형식
                reference = {
                    "id": ref_id,
                    "text": f"[Web] {source.get('title', '제목 없음')}. {source.get('url', source.get('source', 'unknown'))}. Accessed {datetime.now().strftime('%Y-%m-%d')}."
                }
                
            elif source_type == "local":
                # 내부 문서 형식
                reference = {
                    "id": ref_id,
                    "text": f"[Local] {source.get('title', '제목 없음')}. Internal Document. {source.get('source', 'unknown')}. {source.get('year', '')}"
                }
                
            elif source_type == "s3":
                # S3 문서 형식
                reference = {
                    "id": ref_id,
                    "text": f"[S3] {source.get('title', '제목 없음')}. Organization Document. {source.get('source', 'unknown')}. {source.get('year', '')}"
                }
                
            elif source_type == "medgemma":
                # AI 모델 참조 형식
                reference = {
                    "id": ref_id,
                    "text": f"[MedGemma] Medical AI Model Analysis. Generated {datetime.now().strftime('%Y-%m-%d')}."
                }
                
            else:
                # 기타 소스 형식
                reference = {
                    "id": ref_id,
                    "text": f"[{source_type.upper()}] {source.get('title', '제목 없음')}. {source.get('source', 'unknown')}"
                }
            
            references.append(reference)
        
        return references
    
    def _build_formatted_answer(self, answer: str, sources_info: Dict[str, Any], 
                              references: List[Dict[str, Any]], hallucination_attempts: int, 
                              target_language: str, original_question: str = None) -> str:
        """최종 포맷된 답변 구성"""
        
        # 답변에 이미 포맷이 적용되어 있는지 확인
        if "**SUMMARY**" in answer or "**REFERENCES**" in answer:
            # 이미 포맷이 적용된 경우, 참고문헌 섹션만 대체
            if "**REFERENCES**" in answer:
                # 기존 참고문헌 섹션 제거
                answer = re.sub(r'\*\*REFERENCES\*\*[\s\S]*$', '', answer).strip()
            
            formatted_answer = answer
            
            # 참고문헌 섹션 추가
            ref_title = "**REFERENCES**" if target_language == "english" else "**참고문헌**"
            formatted_answer += f"\n\n{ref_title}\n"
            for ref in references:
                formatted_answer += f"{ref['id']}. {ref['text']}\n"
        else:
            # 포맷이 적용되지 않은 경우, 전체 포맷 적용
            formatted_answer = answer.strip()
            
            # 참고문헌 섹션 추가
            ref_title = "**REFERENCES**" if target_language == "english" else "**참고문헌**"
            formatted_answer += f"\n\n{ref_title}\n"
            for ref in references:
                formatted_answer += f"{ref['id']}. {ref['text']}\n"
        
        # 품질 검증 정보 추가
        if hallucination_attempts > 1:
            if target_language == "korean":
                formatted_answer += f"\n\n*이 정보는 {hallucination_attempts}회 검증 과정을 거쳤습니다.*"
            else:
                formatted_answer += f"\n\n*This information has undergone {hallucination_attempts} verification checks.*"
        
        # 소스 유형 사용 통계 추가
        if sources_info["total_count"] > 0:
            source_stats = []
            breakdown = sources_info["breakdown"]
            
            if target_language == "korean":
                if breakdown.get("pubmed", 0) > 0:
                    source_stats.append(f"학술 논문 {breakdown['pubmed']}건")
                if breakdown.get("bedrock_kb", 0) > 0:
                    source_stats.append(f"전문 지식베이스 {breakdown['bedrock_kb']}건")
                if breakdown.get("local", 0) > 0:
                    source_stats.append(f"내부 문서 {breakdown['local']}건")
                if breakdown.get("s3", 0) > 0:
                    source_stats.append(f"기관 문서 {breakdown['s3']}건")
                if breakdown.get("medgemma", 0) > 0:
                    source_stats.append(f"의료 AI 추론 {breakdown['medgemma']}건")
                if breakdown.get("tavily", 0) > 0:
                    source_stats.append(f"웹 자료 {breakdown['tavily']}건")
                
                if source_stats:
                    formatted_answer += f"\n\n*정보 출처: {', '.join(source_stats)}*"
            else:
                if breakdown.get("pubmed", 0) > 0:
                    source_stats.append(f"{breakdown['pubmed']} academic papers")
                if breakdown.get("bedrock_kb", 0) > 0:
                    source_stats.append(f"{breakdown['bedrock_kb']} knowledge base entries")
                if breakdown.get("local", 0) > 0:
                    source_stats.append(f"{breakdown['local']} internal documents")
                if breakdown.get("s3", 0) > 0:
                    source_stats.append(f"{breakdown['s3']} organizational documents")
                if breakdown.get("medgemma", 0) > 0:
                    source_stats.append(f"{breakdown['medgemma']} medical AI inferences")
                if breakdown.get("tavily", 0) > 0:
                    source_stats.append(f"{breakdown['tavily']} web sources")
                
                if source_stats:
                    formatted_answer += f"\n\n*Source breakdown: {', '.join(source_stats)}*"
        
        # 의료 면책 조항 추가
        if target_language == "korean":
            formatted_answer += "\n\n*면책 조항: 이 정보는 의학 참고 자료로 제공되며, 특정 환자의 진단이나 치료를 대체하지 않습니다. 환자 관리에 관한 최종 결정은 담당 의료진의 판단에 따라야 합니다.*"
        else:
            formatted_answer += "\n\n*Disclaimer: This information is provided as a medical reference and does not substitute for the clinical judgment required for the diagnosis or treatment of any specific patient. Final decisions regarding patient care should be made by the treating healthcare provider.*"
        
        return formatted_answer
    
    def format_for_display(self, formatted_output: Dict[str, Any]) -> str:
        """사용자 표시용 최종 텍스트 형태로 변환"""
        return formatted_output.get("main_answer", "답변을 생성할 수 없습니다.")