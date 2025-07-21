# components/output_formatter.py
from typing import Dict, Any, List, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from datetime import datetime
import re
from prompts import system_prompts

class OutputFormatter:
    """의료 전문가용 상세 답변 포맷터 - LLM 사용 버전"""
    
    def __init__(self, llm: ChatOpenAI):
        """
        출력 포맷터 초기화
        
        Args:
            llm: ChatOpenAI 인스턴스
        """
        self.llm = llm
        print("📝 의료 전문가용 출력 포맷터 (LLM 버전) 초기화 완료")
        
        # 포맷팅 체인 설정
        self._setup_formatting_chain()
    
    def _setup_formatting_chain(self):
        """포맷팅 체인 설정"""
        # OUTPUT_FORMATTER 프롬프트 가져오기
        formatter_prompt_content = system_prompts.get("OUTPUT_FORMATTER")
        
        if not formatter_prompt_content:
            print("⚠️ OUTPUT_FORMATTER 프롬프트를 찾을 수 없습니다. 기본 프롬프트 사용.")
            formatter_prompt_content = """You are a medical output formatter. 
            Format the given medical information in a clear, structured manner suitable for healthcare professionals."""
        
        # 프롬프트 템플릿 구성
        self.formatting_prompt = ChatPromptTemplate.from_messages([
            ("system", formatter_prompt_content),
            ("human", """Please format the following medical information:

Raw Answer: {answer}
Question: {question}
Urgency Level: {urgency_level}
Device Type: {device_type}
Clinical Setting: {clinical_setting}
Sources Summary: {sources_summary}
References: {references_text}

Apply appropriate formatting based on the urgency level and device type.""")
        ])
        
        # 포맷팅 체인 구성
        self.formatting_chain = self.formatting_prompt | self.llm | StrOutputParser()
    
    def format_medical_answer(self, 
                            question: str, 
                            answer: str, 
                            source_categorized_docs: Dict[str, List[Document]],
                            conversation_history: List[Dict] = None,
                            hallucination_attempts: int = 1,
                            original_question: str = None,
                            urgency_level: str = None,
                            device_type: str = None,
                            clinical_setting: str = None) -> Dict[str, Any]:
        """
        의료 답변을 전문가용 포맷으로 구성 - LLM 사용
        """
        print("📝 의학 전문가용 답변 포맷팅 (LLM)")
        
        # 소스 정보 구성
        sources_info = self._build_sources_info(source_categorized_docs)
        
        # 참고문헌 목록 생성
        references = self._build_references_list(sources_info["source_list"])
        
        # 긴급도 자동 감지 (제공되지 않은 경우)
        if not urgency_level:
            urgency_level = self._detect_urgency_from_content(question, answer)
        
        # 기본값 설정
        device_type = device_type or "desktop"
        clinical_setting = clinical_setting or "outpatient"
        
        # LLM을 사용한 포맷팅
        try:
            # 소스 요약 생성
            sources_summary = self._create_sources_summary(sources_info)
            
            # 참고문헌 텍스트 생성
            references_text = self._create_references_text(references)
            
            # LLM 포맷팅 호출
            formatted_answer = self.formatting_chain.invoke({
                "answer": answer,
                "question": question,
                "urgency_level": urgency_level,
                "device_type": device_type,
                "clinical_setting": clinical_setting,
                "sources_summary": sources_summary,
                "references_text": references_text
            })
            
            # 품질 검증 정보 추가 (LLM이 누락한 경우)
            if hallucination_attempts > 1 and "검증" not in formatted_answer:
                formatted_answer = self._add_quality_info(formatted_answer, hallucination_attempts)
            
            print("  ✅ LLM 포맷팅 완료")
            
        except Exception as e:
            print(f"  ❌ LLM 포맷팅 실패: {str(e)}")
            # 폴백: 기본 포맷팅 사용
            formatted_answer = self._fallback_formatting(
                answer, references, sources_info, hallucination_attempts
            )
        
        return {
            "main_answer": formatted_answer,
            "sources_used": sources_info["source_list"],
            "total_sources": sources_info["total_count"],
            "references": references,
            "urgency_level": urgency_level,
            "device_type": device_type,
            "metadata": {
                "question": question,
                "generated_at": datetime.now().isoformat(),
                "hallucination_checks": hallucination_attempts,
                "source_breakdown": sources_info["breakdown"],
                "clinical_setting": clinical_setting,
                "formatted_by": "llm"
            }
        }
    
    def _create_sources_summary(self, sources_info: Dict[str, Any]) -> str:
        """소스 정보 요약 생성"""
        breakdown = sources_info["breakdown"]
        summary_parts = []
        
        source_names = {
            "pubmed": "PubMed 학술논문",
            "bedrock_kb": "의학 지식베이스",
            "local": "기관 내부문서",
            "s3": "임상 자료",
            "medgemma": "MedGemma AI",
            "tavily": "웹 검색결과"
        }
        
        for source_type, count in breakdown.items():
            if count > 0:
                name = source_names.get(source_type, source_type)
                summary_parts.append(f"{name} {count}건")
        
        return f"총 {sources_info['total_count']}개 소스 사용: " + ", ".join(summary_parts)
    
    def _create_references_text(self, references: List[Dict[str, Any]]) -> str:
        """참고문헌 텍스트 생성"""
        if not references:
            return "참고문헌 없음"
        
        ref_lines = []
        for ref in references:
            ref_lines.append(f"{ref['id']}. {ref['text']}")
        
        return "\n".join(ref_lines)
    
    def _detect_urgency_from_content(self, question: str, answer: str) -> str:
        """내용에서 긴급도 자동 감지"""
        combined_text = f"{question} {answer}".lower()
        
        # 긴급도 키워드 매핑
        emergency_keywords = ["응급", "즉시", "emergency", "stat", "critical", "생명"]
        urgent_keywords = ["긴급", "urgent", "빠른", "신속"]
        preventive_keywords = ["예방", "검진", "screening", "prevention"]
        
        if any(keyword in combined_text for keyword in emergency_keywords):
            return "emergency"
        elif any(keyword in combined_text for keyword in urgent_keywords):
            return "urgent"
        elif any(keyword in combined_text for keyword in preventive_keywords):
            return "preventive"
        else:
            return "routine"
    
    def _fallback_formatting(self, answer: str, references: List[Dict[str, Any]], 
                           sources_info: Dict[str, Any], hallucination_attempts: int) -> str:
        """LLM 실패 시 기본 포맷팅"""
        print("  🔄 기본 포맷팅 사용")
        
        formatted_parts = [answer.strip()]
        
        # 참고문헌 추가
        if references:
            ref_section = "\n\n**📚 참고문헌 (REFERENCES)**"
            for ref in references:
                ref_section += f"\n{ref['id']}. {ref['text']}"
            formatted_parts.append(ref_section)
        
        # 품질 정보
        if hallucination_attempts > 1:
            formatted_parts.append(f"\n✅ *{hallucination_attempts}회 정확성 검증 완료*")
        
        # 출처 통계
        source_stats = self._build_source_stats_text(sources_info["breakdown"])
        if source_stats:
            formatted_parts.append(f"📊 *정보 출처: {source_stats}*")
        
        # 면책조항
        formatted_parts.append("*의학 참고 자료. 임상 판단은 담당 의료진이 결정.*")
        
        return "\n".join(formatted_parts)
    
    def _add_quality_info(self, answer: str, hallucination_attempts: int) -> str:
        """품질 검증 정보 추가"""
        quality_text = f"\n\n✅ *{hallucination_attempts}회 정확성 검증 완료*"
        
        # 적절한 위치에 추가
        if "참고문헌" in answer or "REFERENCES" in answer:
            # 참고문헌 섹션 다음에 추가
            parts = answer.split("참고문헌")
            if len(parts) > 1:
                # 참고문헌 섹션 끝 찾기
                ref_end = parts[1].find("\n\n")
                if ref_end > 0:
                    parts[1] = parts[1][:ref_end] + quality_text + parts[1][ref_end:]
                else:
                    parts[1] = parts[1] + quality_text
                return "참고문헌".join(parts)
        
        # 그 외의 경우 끝에 추가
        return answer + quality_text
    
    def _build_source_stats_text(self, breakdown: Dict[str, int]) -> str:
        """출처 통계 텍스트 생성"""
        source_names = {
            "pubmed": "학술논문",
            "bedrock_kb": "의학KB",
            "local": "기관문서",
            "s3": "임상자료",
            "medgemma": "AI분석",
            "tavily": "웹자료"
        }
        
        stats = []
        for source_type, count in breakdown.items():
            if count > 0:
                name = source_names.get(source_type, source_type)
                stats.append(f"{name} {count}건")
        
        return ", ".join(stats)
    
    # 기존 헬퍼 메서드들은 그대로 유지
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
                source_info = {
                    "type": source_type,
                    "source": doc.metadata.get("source", "unknown"),
                    "title": doc.metadata.get("title", "제목 없음"),
                    "authors": doc.metadata.get("authors", ""),
                    "year": doc.metadata.get("year", ""),
                    "journal": doc.metadata.get("journal", ""),
                    "url": doc.metadata.get("url", ""),
                    "doi": doc.metadata.get("doi", ""),
                    "similarity_score": doc.metadata.get("similarity_score", 0)
                }
                source_list.append(source_info)
        
        source_list.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return {
            "source_list": source_list,
            "breakdown": breakdown,
            "total_count": total_count
        }
    
    def _build_references_list(self, source_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """참고문헌 목록 생성 - URL 링크 포함"""
        references = []
        seen_titles = set()
        
        for i, source in enumerate(source_list):
            title = source.get("title", "")
            if title in seen_titles and title != "제목 없음":
                continue
            seen_titles.add(title)
            
            ref_id = len(references) + 1
            source_type = source["type"]
            
            # URL 정보 추출 (모든 소스에 공통 적용)
            url = source.get("url", "")
            source_path = source.get("source", "")
            
            # 소스 타입별 참고문헌 포맷
            if source_type == "pubmed":
                # PubMed 처리 개선
                authors = source.get("authors", "")
                if isinstance(authors, list):
                    authors = ", ".join(authors[:3])
                    if len(source.get("authors", [])) > 3:
                        authors += " et al."
                
                year = source.get("year", "")
                journal = source.get("journal", "")
                pmid = source.get("pmid", "")
                
                # PubMed URL 생성
                if pmid and not url:
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                
                ref_text = f"[PUBMED] {authors}. {source.get('title', '제목 없음')}."
                if journal:
                    ref_text += f" {journal}."
                if year:
                    ref_text += f" {year}."
                if url:
                    ref_text += f" URL: {url}"
                
                reference = {"id": ref_id, "text": ref_text, "url": url}
                
            elif source_type == "bedrock_kb":
                # Bedrock KB 개선
                # S3 링크 생성 (예시 - 실제 구현은 환경에 맞게 조정 필요)
                s3_bucket = "arn:aws:s3:::aws-medical-chatbot"  # 실제 버킷 이름으로 변경 필요
                
                if source_path and not url:
                    # S3 경로 정리 및 URL 생성
                    s3_key = source_path
                    if s3_key.startswith("/"):
                        s3_key = s3_key[1:]
                    
                    # 일반 S3 경로 생성 (비공개 버킷의 경우 서명된 URL 생성 로직 필요)
                    url = f"s3://{s3_bucket}/{s3_key}"
                
                title = source.get("title", "")
                if not title or title == "제목 없음":
                    title = source_path.split("/")[-1] if "/" in source_path else source_path
                    title = title.replace("_", " ").replace(".pdf", "").replace(".PDF", "")
                
                ref_text = f"[BEDROCK_KB] {title}"
                if url:
                    ref_text += f" URL: {url}"
                
                reference = {
                    "id": ref_id,
                    "text": ref_text,
                    "url": url
                }
                    
            elif source_type == "tavily":
                # Web 검색 결과 - 직접 URL 포함
                url = url or source.get("source", "")
                
                ref_text = f"[WEB] {source.get('title', '제목 없음')}."
                if url:
                    ref_text += f" URL: {url}"
                ref_text += f" Accessed {datetime.now().strftime('%Y-%m-%d')}."
                
                reference = {
                    "id": ref_id,
                    "text": ref_text,
                    "url": url
                }
                
            elif source_type == "medgemma":
                # MedGemma 소스 추가
                model_name = source.get("model_name", "MedGemma")
                source_info = source.get("source_info", "")
                
                ref_text = f"[MEDGEMMA] AI 분석: {model_name}"
                if source_info:
                    ref_text += f" - {source_info}"
                if url:
                    ref_text += f" URL: {url}"
                
                reference = {
                    "id": ref_id,
                    "text": ref_text,
                    "url": url
                }
                
            else:
                # 기타 소스
                ref_text = f"[{source_type.upper()}] {source.get('title', '제목 없음')}. {source_path}"
                if url:
                    ref_text += f" URL: {url}"
                
                reference = {
                    "id": ref_id,
                    "text": ref_text,
                    "url": url
                }
            
            references.append(reference)
        
        return references    
    def format_for_display(self, formatted_output: Dict[str, Any]) -> str:
        """사용자 표시용 최종 텍스트 형태로 변환"""
        answer = formatted_output.get("main_answer", "답변을 생성할 수 없습니다.")
        
        # 줄바꿈 자동 수정
        answer = self._fix_formatting_issues(answer)
        
        return answer

    def _fix_formatting_issues(self, text: str) -> str:
        """포맷팅 문제 자동 수정"""
        import re
        
        # 1. 섹션 제목 다음 줄바꿈 추가
        # **제목** • → **제목**\n\n•
        text = re.sub(r'(\*\*[^*]+\*\*)\s*•', r'\1\n\n•', text)
        
        # 2. 섹션 제목 다음 일반 텍스트도 줄바꿈
        # **제목** 텍스트 → **제목**\n\n텍스트
        text = re.sub(r'(\*\*[^*]+\*\*)\s+([^•\n*])', r'\1\n\n\2', text)
        
        # 3. 연속된 bullet points 분리
        # • 항목1 • 항목2 → • 항목1\n• 항목2
        while ' • ' in text:
            text = text.replace(' • ', '\n• ')
        
        # 4. 숫자 목록 줄바꿈
        # 1. 항목 2. 항목 → 1. 항목\n2. 항목
        text = re.sub(r'(\d+\.\s+[^0-9\n]+)\s+(\d+\.)', r'\1\n\2', text)
        
        # 5. 주요 섹션 사이 간격 확보
        sections = ['진단 접근', '치료 계획', '모니터링', '참고문헌', 
                    'DIAGNOSTIC APPROACH', 'TREATMENT PLAN', 'MONITORING', 'REFERENCES']
        
        for section in sections:
            # 섹션 앞에 빈 줄 추가
            text = text.replace(f'\n**{section}**', f'\n\n**{section}**')
            text = text.replace(f'**{section}**', f'\n\n**{section}**')
        
        # 6. 중복 빈 줄 제거
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 7. 들여쓰기 수정
        lines = text.split('\n')
        fixed_lines = []
        in_numbered_list = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # 숫자 목록 감지
            if re.match(r'^\d+\.', stripped):
                in_numbered_list = True
                fixed_lines.append(stripped)
            # 숫자 목록 내의 bullet
            elif in_numbered_list and stripped.startswith('•'):
                fixed_lines.append('   ' + stripped)
            # 일반 라인
            else:
                if not stripped.startswith('•') and not re.match(r'^\d+\.', stripped):
                    in_numbered_list = False
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines).strip()