# components/integrator.py (리팩토링된 버전)
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from prompts import system_prompts

class Integrator:
    """다중 소스 정보 통합 담당 클래스 (가중치 적용)"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        
        # 소스별 신뢰도 가중치
        self.source_weights = {
            "pubmed": 1.0,     # 최고 신뢰도 (학술 논문)
            "bedrock_kb": 0.9, # AWS Bedrock KB (높은 신뢰도)
            "medgemma": 0.9,   # 높은 신뢰도 (의료 특화 AI)
            "rag": 0.8,        # 높은 신뢰도 (큐레이션된 데이터)
            "web": 0.6         # 중간 신뢰도 (웹 검색)
        }

        self._setup_integration_chain()
    
    def _setup_integration_chain(self):
        """정보 통합 체인 설정"""
        # 하드코딩된 프롬프트 대신 system_prompts 사용
        # 가중치 변수를 템플릿에 전달하여 동적 프롬프트 생성
        self.integration_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompts.format("INTEGRATOR", 
                pubmed_weight=self.source_weights.get("pubmed", 1.0),
                bedrock_weight=self.source_weights.get("bedrock_kb", 0.9),
                rag_weight=self.source_weights.get("rag", 0.8),
                web_weight=self.source_weights.get("web", 0.6),
                medgemma_weight=self.source_weights.get("medgemma", 0.9)
            )),
            ("human", """Question: {question}

        Sources with weights:
        {weighted_content}

        Provide integrated medical answer with clear source citations for each piece of information:"""),
        ])
        
        self.integration_chain = self.integration_prompt | self.llm | StrOutputParser()
    
    def integrate_answers(self, question: str, source_categorized_docs: Dict[str, List[Document]]) -> str:
        """다중 소스 정보를 가중치 적용하여 통합"""
        print("==== [INTEGRATE WITH WEIGHTS] ====")
        
        if not source_categorized_docs:
            return "관련 정보를 찾을 수 없어 답변을 생성할 수 없습니다."
        
        # 가중치 적용된 내용 구성
        weighted_content = self._build_weighted_content(source_categorized_docs)
        
        try:
            integrated_answer = self.integration_chain.invoke({
                "question": question,
                "weighted_content": weighted_content
            })
            
            # 출처 표기 형식 개선
            enhanced_answer = self._enhance_citations(integrated_answer)
            
            print(f"  ✅ 소스 통합 완료 ({len(source_categorized_docs)}개 소스)")
            return enhanced_answer
            
        except Exception as e:
            print(f"  ❌ 통합 실패: {str(e)}")
            return self._fallback_integration(source_categorized_docs)

    def _enhance_citations(self, answer: str) -> str:
        """출처 표기 형식 개선"""
        import re
        
        # 출처 표기 강조 및 일관성 유지
        # [SOURCE_TYPE: specific source] 형식을 일관되게 변환
        
        # 정규식 패턴
        citation_pattern = r'\[((?:PubMed|Web|Bedrock_KB|RAG|S3|MedGemma)[^]]*)\]'
        
        # 출처 표기 강조
        def citation_replacer(match):
            citation = match.group(1)
            return f'【{citation}】'
        
        # 정규식으로 출처 표기 변환
        enhanced = re.sub(citation_pattern, citation_replacer, answer)
        
        # 출처가 없는 문장에 대한 안내 추가
        if '【' not in enhanced:
            enhanced += "\n\n(⚠️ 참고: 이 답변은 제공된 정보를 바탕으로 생성되었으나, 구체적인 출처를 표기하지 않았습니다. 정확한 의료 정보는 의료 전문가와 상담하세요.)"
        
        return enhanced
    
    def _build_weighted_content(self, categorized_docs: Dict[str, List[Document]]) -> str:
        """소스별 가중치를 적용한 내용 구성"""
        content_parts = []
        
        for source_type, docs in categorized_docs.items():
            if not docs:
                continue
                
            weight = self.source_weights.get(source_type, 0.5)
            
            content_parts.append(f"\n=== {source_type.upper()} SOURCES (신뢰도: {weight}) ===")
            
            for i, doc in enumerate(docs):
                # 소스 유형별 출처 정보 추출
                source_info = self._extract_source_info(source_type, doc)
                content = doc.page_content[:300]  # 300자 제한
                
                # 출처 정보 포함
                content_parts.append(f"{i+1}. [{source_info}] {content}")
        
        return "\n".join(content_parts)

    def _extract_source_info(self, source_type: str, doc: Document) -> str:
        """문서 유형별 출처 정보 추출"""
        metadata = doc.metadata or {}
        
        if source_type == "pubmed":
            # PubMed 논문 정보
            authors = metadata.get("authors", [])
            author_text = f"{authors[0]} 외" if authors and len(authors) > 1 else ", ".join(authors) if authors else "Unknown"
            year = metadata.get("year", "")
            journal = metadata.get("journal", "")
            return f"PubMed: {author_text} ({year}), {journal}"
        
        elif source_type == "web":
            # 웹 출처 정보
            source = metadata.get("source", "")
            # URL에서 도메인만 추출
            import re
            domain = ""
            if isinstance(source, str) and "://" in source:
                match = re.search(r'://([^/]+)', source)
                if match:
                    domain = match.group(1)
                    # www. 제거
                    domain = re.sub(r'^www\.', '', domain)
            
            return f"Web: {domain or source or 'Unknown website'}"
        
        elif source_type == "bedrock_kb":
            # Bedrock KB 문서 정보 - 더 구체적인 정보
            title = metadata.get("title", "")
            doc_id = metadata.get("document_id", "")
            category = metadata.get("category", "")
            
            # Document만 표시되는 경우 내용에서 제목 추출 시도
            if not title and not doc_id:
                content = doc.page_content or ""
                # 첫 줄이나 첫 10단어를 제목으로 사용
                first_line = content.split('\n')[0] if '\n' in content else ""
                content_preview = first_line[:50] if first_line else " ".join(content.split()[:7])
                
                if content_preview:
                    return f"Bedrock KB: {content_preview}..."
            
            return f"Bedrock KB: {title or doc_id or category or 'Medical document'}"
        
        elif source_type == "s3":
            # S3 문서 정보
            path = metadata.get("source", "")
            title = metadata.get("title", "")
            # 경로에서 파일명만 추출
            if isinstance(path, str):
                import os
                filename = os.path.basename(path)
            else:
                filename = ""
            
            return f"S3: {title or filename or 'Document'}"
        
        elif source_type == "rag":
            # 내부 RAG 문서 정보
            source = metadata.get("source", "")
            title = metadata.get("title", "")
            category = metadata.get("category", "")
            
            # 소스에서 파일명만 추출
            if isinstance(source, str):
                import os
                filename = os.path.basename(source)
            else:
                filename = ""
            
            return f"RAG: {title or filename or category or 'Document'}"
        
        elif source_type == "medgemma":
            # MedGemma 정보
            model = metadata.get("model_name", "MedGemma")
            return f"MedGemma: {model}"
        
        # 기본 출처 정보
        return f"{source_type}: {metadata.get('source', 'Unknown')}"
    
    def _fallback_integration(self, categorized_docs: Dict[str, List[Document]]) -> str:
        """통합 실패 시 폴백 방법"""
        print("  🔄 기본 통합 방식 사용")
        
        # 가장 신뢰도 높은 소스부터 사용
        for source_type in ["pubmed", "medgemma", "rag"]:
            if source_type in categorized_docs and categorized_docs[source_type]:
                docs = categorized_docs[source_type]
                weight = self.source_weights[source_type]
                
                return f"""다음 정보를 바탕으로 답변드립니다 (신뢰도: {weight}):

{docs[0].page_content[:500]}

이 정보는 {source_type} 소스에서 가져온 것입니다. 
정확한 의료 정보를 위해서는 의료 전문가와 상담하시기 바랍니다."""
        
        return "죄송합니다. 신뢰할 수 있는 정보를 찾을 수 없습니다."