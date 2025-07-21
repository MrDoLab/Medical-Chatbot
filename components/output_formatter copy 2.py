# components/output_formatter.py
from typing import Dict, Any, List, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from datetime import datetime
import re
import os
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
        
        # 링크 보존 및 리스트 번호 유지 지침 추가
        formatter_prompt_content += """

        IMPORTANT FORMATTING INSTRUCTIONS:
        1. Do not modify or remove any links, citations, or references in the content.
        2. Preserve all markdown links in the format [text](url) or 【text】(url).
        3. Keep all reference numbers exactly as provided, especially numerical citations like [1,2].
        4. Maintain numbered lists in the correct sequential order (1, 2, 3, etc.).
        5. Do not reset list numbering within the same section - use proper incremental numbering.
        6. If a section has multiple numbered items, ensure they are numbered sequentially (1, 2, 3, etc.).
        7. When formatting diagnostic or treatment sections, maintain the hierarchical structure.
        8. Preserve all original citations and reference formats exactly as they appear in the text.
        """
        
        # 프롬프트 템플릿 구성
        self.formatting_prompt = ChatPromptTemplate.from_messages([
            ("system", formatter_prompt_content),
            ("human", """Please format the following medical information, paying special attention to maintaining numbered lists, citations, and links:

Raw Answer: {answer}
Question: {question}
Urgency Level: {urgency_level}
Device Type: {device_type}
Clinical Setting: {clinical_setting}
Sources Summary: {sources_summary}
References: {references_text}

Apply appropriate formatting based on the urgency level and device type.
Make sure to preserve all links, citations and references exactly as they appear in the original text.
Ensure numbered lists use sequential numbering (1, 2, 3) instead of restarting at 1 for each item.""")
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
        
        # 참고문헌 섹션이 이미 있는지 확인
        has_references_section = "## 참고문헌" in answer or "## 정보 출처" in answer or "**참고문헌**" in answer
        
        # 출처가 없는 경우 경고 메시지 제거
        if '【' in answer or '[' in answer and ']' in answer and any(str(i) in answer for i in range(10)):
            # 인용이 있으면 경고 메시지 제거
            answer = re.sub(r'\n\n\(⚠️ 참고: 이 답변은 제공된 정보를 바탕으로 생성되었으나, 구체적인 출처를 표기하지 않았습니다.*?\)', '', answer)
        
        # LLM을 사용한 포맷팅
        try:
            # 소스 요약 생성
            sources_summary = self._create_sources_summary(sources_info)
            
            # 참고문헌 텍스트 생성 (기존 섹션이 없는 경우에만)
            if not has_references_section:
                references_text = self._create_references_text(references)
                
                # 참고문헌 섹션 직접 추가
                answer += self._format_references_section(references)
            else:
                # 이미 참고문헌 섹션이 있으면 빈 텍스트 전달
                references_text = ""
            
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
            
            # 리스트 번호 수정
            formatted_answer = self._fix_list_numbering(formatted_answer)
            
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
    
    def _format_references_section(self, references: List[Dict[str, Any]]) -> str:
        """참고문헌 섹션 직접 포맷팅"""
        if not references:
            return ""
        
        ref_section = "\n\n## 참고문헌\n"
        for ref in references:
            ref_id = ref["id"]
            ref_text = ref["text"]
            url = ref.get("url")
            
            # URL이 있으면 마크다운 링크로 표시
            if url:
                # 링크 텍스트가 있는지 확인
                if "[링크]" in ref_text:
                    # "[링크]"를 마크다운 링크로 변환
                    ref_text = ref_text.replace("[링크]", f"[링크]({url})")
                else:
                    # 링크가 없으면 끝에 추가
                    ref_text += f" [링크]({url})"
            
            ref_section += f"{ref_id}. {ref_text}\n"
            
        return ref_section
    
    def _fix_list_numbering(self, text: str) -> str:
        """리스트 번호 순서 수정"""
        # 각 섹션 찾기
        section_pattern = r'(\*\*[^*]+\*\*\s*\n+)((?:\d+\.\s+[^\n]+\n+)+)'
        
        def fix_section_numbering(match):
            section_header = match.group(1)
            list_content = match.group(2)
            
            # 리스트 항목 찾기
            list_items = re.findall(r'\d+\.\s+([^\n]+)', list_content)
            
            # 번호 재지정
            fixed_list = ""
            for i, item in enumerate(list_items, 1):
                fixed_list += f"{i}. {item}\n"
            
            return section_header + fixed_list
        
        # 섹션별로 리스트 번호 수정
        return re.sub(section_pattern, fix_section_numbering, text)
    
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
            ref_id = ref["id"]
            ref_text = ref["text"]
            url = ref.get("url")
            
            if url:
                # URL이 있으면 마크다운 링크로 표시
                if "[링크]" in ref_text:
                    ref_text = ref_text.replace("[링크]", f"[링크]({url})")
                else:
                    ref_text += f" [링크]({url})"
            
            ref_lines.append(f"{ref_id}. {ref_text}")
        
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
        if references and "## 참고문헌" not in answer and "## 정보 출처" not in answer:
            ref_section = "\n\n## 참고문헌\n"
            for ref in references:
                ref_id = ref["id"]
                ref_text = ref["text"]
                url = ref.get("url")
                
                if url:
                    # URL이 있으면 마크다운 링크로 표시
                    if "[링크]" in ref_text:
                        ref_text = ref_text.replace("[링크]", f"[링크]({url})")
                    else:
                        ref_text += f" [링크]({url})"
                
                ref_section += f"{ref_id}. {ref_text}\n"
            
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
                    "s3_location": doc.metadata.get("s3_location", ""),
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
            s3_location = source.get("s3_location", "")
            
            # S3 경로 처리 (s3_location이 있으면 우선 사용)
            s3_uri = s3_location if s3_location else (
                source_path if isinstance(source_path, str) and source_path.startswith("s3://") else None
            )
            
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
                    ref_text += f" [링크]"  # URL은 링크 텍스트로만 표시
                
                reference = {"id": ref_id, "text": ref_text, "url": url}
                
            elif source_type == "bedrock_kb":
                # Bedrock KB 개선
                title = source.get("title", "")
                
                # 페이지 정보 추출 (예: p164-326)
                page_info = ""
                if isinstance(source_path, str):
                    # 페이지 패턴 검색 (p숫자-숫자)
                    page_match = re.search(r'p\d+-\d+', source_path)
                    if page_match:
                        page_info = page_match.group(0)
                    else:
                        # 파일명 추출 시도
                        parts = source_path.split("/")
                        filename = parts[-1] if parts else ""
                        # 파일명이 있으면 사용
                        if filename:
                            clean_filename = filename.replace("_", " ").replace(".pdf", "").replace(".PDF", "")
                            if not title or title == "제목 없음":
                                title = clean_filename
                
                # S3 URI가 있으면 폴더 정보 추출
                folder_info = ""
                if s3_uri and isinstance(s3_uri, str):
                    parts = s3_uri.split("/")
                    if len(parts) > 3:
                        # 최대 2단계 폴더 정보 추출
                        folder_info = "/".join(parts[-3:-1])
                
                ref_text = f"[BEDROCK_KB] {title}"
                if page_info:
                    ref_text += f", {page_info}"
                if folder_info:
                    ref_text += f" (위치: {folder_info})"
                
                # 공식 콘솔 URL 생성 예시 (실제 환경에 맞게 조정 필요)
                console_url = None
                if s3_uri:
                    # AWS S3 콘솔 URL 예시 (실제 구현은 환경에 맞게 조정 필요)
                    # console_url = f"https://s3.console.aws.amazon.com/s3/object/{s3_uri[5:]}"
                    pass
                
                reference = {"id": ref_id, "text": ref_text, "url": console_url}
                    
            elif source_type == "tavily" or source_type == "web":
                # Web 검색 결과 - 간결한 링크 표시
                url = url or source.get("source", "")
                title = source.get("title", "")
                domain = source.get("domain", "")
                
                # 도메인 추출
                if not domain and url:
                    match = re.search(r'://([^/]+)', url)
                    domain = match.group(1) if match else ""
                
                ref_text = f"[WEB] {title or domain or '웹 문서'}"
                if url:
                    ref_text += f" - [링크]"  # URL은 링크 텍스트로만 표시
                
                reference = {"id": ref_id, "text": ref_text, "url": url}
                
            elif source_type == "medgemma":
                # MedGemma 소스 - 링크 추가
                model_name = source.get("model_name", "MedGemma")
                source_info = source.get("source_info", "")
                
                ref_text = f"[MEDGEMMA] AI 분석: {model_name}"
                if source_info:
                    ref_text += f" - {source_info}"
                
                # MedGemma 공식 링크 추가
                model_link = "https://blog.google/technology/developers/gemma-open-models/" if "gemma" in str(model_name).lower() else None
                if model_link:
                    ref_text += f" - [링크]"
                
                reference = {"id": ref_id, "text": ref_text, "url": model_link}
                
            else:
                # 기타 소스 - 간결한 표시
                ref_text = f"[{source_type.upper()}] {source.get('title', '제목 없음')}"
                
                # 소스 경로 간소화
                if source_path and isinstance(source_path, str):
                    filename = os.path.basename(source_path)
                    ref_text += f" - {filename}"
                
                if url:
                    ref_text += f" - [링크]"
                
                reference = {"id": ref_id, "text": ref_text, "url": url}
            
            references.append(reference)
        
        return references
    
    def format_for_display(self, formatted_output: Dict[str, Any]) -> str:
        """사용자 표시용 최종 텍스트 형태로 변환"""
        answer = formatted_output.get("main_answer", "답변을 생성할 수 없습니다.")
        
        # 줄바꿈 자동 수정
        answer = self._fix_formatting_issues(answer)
        
        # 링크 형식 확인 및 수정
        answer = self._ensure_markdown_links(answer)
        
        return answer
    
    def _ensure_markdown_links(self, text: str) -> str:
        """마크다운 링크 형식 확인 및 수정"""
        # 【텍스트】(URL) 형식 확인 - 공백이 끼어 있는 경우 수정
        text = re.sub(r'【([^】]+)】\s+\(([^)]+)\)', r'【\1】(\2)', text)
        
        # [텍스트](URL) 형식 확인 - 공백이 끼어 있는 경우 수정
        text = re.sub(r'\[([^\]]+)\]\s+\(([^)]+)\)', r'[\1](\2)', text)
        
        # [링크] 텍스트 찾아서 URL이 있는지 확인 (참고문헌 섹션에서)
        if "## 참고문헌" in text or "## 정보 출처" in text:
            lines = text.split("\n")
            in_references = False
            for i, line in enumerate(lines):
                if "## 참고문헌" in line or "## 정보 출처" in line:
                    in_references = True
                    continue
                
                if in_references and "[링크]" in line and not re.search(r'\[링크\]\([^)]+\)', line):
                    # URL 추출 시도
                    parts = line.split(". ", 1)
                    if len(parts) > 1 and "[" in parts[1]:
                        # 소스 유형 확인
                        if "[PUBMED]" in parts[1]:
                            # PubMed 링크 생성
                            pmid_match = re.search(r'PMID:(\d+)', parts[1])
                            if pmid_match:
                                pmid = pmid_match.group(1)
                                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                                lines[i] = line.replace("[링크]", f"[링크]({url})")
                        elif "[MEDGEMMA]" in parts[1]:
                            # MedGemma 링크
                            lines[i] = line.replace("[링크]", "[링크](https://blog.google/technology/developers/gemma-open-models/)")
            
            text = "\n".join(lines)
        
        return text

    def _fix_formatting_issues(self, text: str) -> str:
        """포맷팅 문제 자동 수정 - 줄바꿈 최소화"""
        import re
        
        # 마크다운 링크 패턴을 보존하기 위해 임시 치환
        # 【텍스트】(URL) 형식 보존
        link_pattern = r'(【[^】]+】)(\([^)]+\))'
        links = {}
        link_count = 0
        
        def save_link(match):
            nonlocal link_count
            placeholder = f"__LINK_PLACEHOLDER_{link_count}__"
            links[placeholder] = match.group(0)
            link_count += 1
            return placeholder
        
        # 링크를 임시 플레이스홀더로 대체
        text = re.sub(link_pattern, save_link, text)
        
        # 일반적인 마크다운 링크도 보존
        md_link_pattern = r'(\[[^\]]+\])(\([^)]+\))'
        text = re.sub(md_link_pattern, save_link, text)
        
        # 1. 섹션 제목 다음 줄바꿈 최소화 - 하나의 줄바꿈만 추가
        text = re.sub(r'(\*\*[^*]+\*\*)\s*•', r'\1\n•', text)
        text = re.sub(r'(\*\*[^*]+\*\*)\s+([^•\n*])', r'\1\n\2', text)
        
        # 2. 연속된 bullet points 분리 - 필요한 경우만
        if ' • ' in text:
            text = re.sub(r' • ', '\n• ', text)
        
        # 3. 숫자 목록 줄바꿈 - 필요한 경우만
        text = re.sub(r'(\d+\.\s+[^0-9\n]+)\s+(\d+\.)', r'\1\n\2', text)
        
        # 4. 주요 섹션 처리 - 줄바꿈 최소화
        sections = ['진단 접근', '치료 계획', '모니터링', '참고문헌', '정보 출처',
                    'DIAGNOSTIC APPROACH', 'TREATMENT PLAN', 'MONITORING', 'REFERENCES']
        
        for section in sections:
            # 이미 줄바꿈이 있는 경우에는 추가 줄바꿈을 넣지 않음
            text = re.sub(r'([^\n])\n\*\*' + section + r'\*\*', r'\1\n**' + section + '**', text)
            # 문서 시작 부분이거나 이미 줄바꿈이 있으면 추가하지 않음
            text = re.sub(r'^([^\n]*)\*\*' + section + r'\*\*', r'\1**' + section + '**', text)
        
        # 5. 연속된 빈 줄 제거 - 최대 한 개의 빈 줄만 허용
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 6. 들여쓰기 수정 - 줄바꿈 추가 없이
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
        
        text = '\n'.join(fixed_lines).strip()
        
        # 7. 리스트 번호 수정 - 번호 순서만 고치고 줄바꿈은 추가하지 않음
        section_pattern = r'(\*\*[^*]+\*\*\s*\n)((?:\d+\.\s+[^\n]+\n+)+)'
        
        def fix_section_numbering(match):
            section_header = match.group(1)
            list_content = match.group(2)
            
            # 리스트 항목 찾기
            list_items = re.findall(r'\d+\.\s+([^\n]+)', list_content)
            
            # 번호 재지정 (줄바꿈 최소화)
            fixed_list = ""
            for i, item in enumerate(list_items, 1):
                if i < len(list_items):
                    fixed_list += f"{i}. {item}\n"
                else:
                    fixed_list += f"{i}. {item}"
            
            return section_header + fixed_list
        
        # 섹션별로 리스트 번호 수정
        text = re.sub(section_pattern, fix_section_numbering, text)
        
        # 마지막에 플레이스홀더를 원래 링크로 복원
        for placeholder, link in links.items():
            text = text.replace(placeholder, link)
        
        return text