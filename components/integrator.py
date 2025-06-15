# components/integrator.py (리팩토링된 버전)
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

class Integrator:
    """다중 소스 정보 통합 담당 클래스 (가중치 적용)"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self._setup_integration_chain()
        
        # 소스별 신뢰도 가중치
        self.source_weights = {
            "pubmed": 1.0,    # 최고 신뢰도 (학술 논문)
            "medgemma": 0.9,  # 높은 신뢰도 (의료 특화 AI)
            "rag": 0.8,       # 높은 신뢰도 (큐레이션된 데이터)
            "web": 0.6        # 중간 신뢰도 (웹 검색)
        }
    
    def _setup_integration_chain(self):
        """정보 통합 체인 설정"""
        self.integration_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a medical information integrator. Combine multiple sources to provide accurate medical answers.

            Source Reliability Guide:
            - PubMed (Weight: 1.0): Peer-reviewed academic papers - highest reliability
            - RAG (Weight: 0.8): Curated medical database - high reliability  
            - Web (Weight: 0.6): General web sources - moderate reliability
            
            Integration Guidelines:
            - Prioritize information by source reliability
            - Synthesize complementary information from multiple sources
            - Note any important contradictions between sources
            - RESPOND IN THE SAME LANGUAGE AS THE USER'S INPUT (Korean for Korean input, English for English input, etc.)
            - Focus on medical accuracy and patient safety"""),
            ("human", """Question: {question}
            
            Sources with weights:
            {weighted_content}
            
            Provide integrated medical answer:"""),
        ])
        
        self.integration_chain = self.integration_prompt | self.llm | StrOutputParser()
    
    def integrate_answers(self, question: str, categorized_docs: Dict[str, List[Document]]) -> str:
        """다중 소스 정보를 가중치 적용하여 통합"""
        print("==== [INTEGRATE WITH WEIGHTS] ====")
        
        if not categorized_docs:
            return "관련 정보를 찾을 수 없어 답변을 생성할 수 없습니다."
        
        # 가중치 적용된 내용 구성
        weighted_content = self._build_weighted_content(categorized_docs)
        
        try:
            integrated_answer = self.integration_chain.invoke({
                "question": question,
                "weighted_content": weighted_content
            })
            
            print(f"  ✅ 소스 통합 완료 ({len(categorized_docs)}개 소스)")
            return integrated_answer
            
        except Exception as e:
            print(f"  ❌ 통합 실패: {str(e)}")
            return self._fallback_integration(categorized_docs)
    
    def _build_weighted_content(self, categorized_docs: Dict[str, List[Document]]) -> str:
        """소스별 가중치를 적용한 내용 구성"""
        content_parts = []
        
        for source_type, docs in categorized_docs.items():
            if not docs:
                continue
                
            weight = self.source_weights.get(source_type, 0.5)
            
            content_parts.append(f"\n=== {source_type.upper()} SOURCES (신뢰도: {weight}) ===")
            
            for i, doc in enumerate(docs):
                source = doc.metadata.get("source", "unknown")
                content = doc.page_content[:300]  # 300자 제한
                content_parts.append(f"{i+1}. [{source}] {content}")
        
        return "\n".join(content_parts)
    
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