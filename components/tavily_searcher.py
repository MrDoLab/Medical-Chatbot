# components/tavily_searcher.py
from typing import List, Dict, Any
from langchain_core.documents import Document
import requests
from datetime import datetime
import os

class TavilySearcher:
    """Tavily API 기반 웹 검색 담당 클래스"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.api_url = "https://api.tavily.com/search"
        
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY가 설정되지 않았습니다!")
            
        # 검색 통계
        self.search_stats = {
            "queries_processed": 0,
            "successful_searches": 0,
            "failed_searches": 0
        }
    
    def search_web(self, query: str, max_results: int = 5) -> List[Document]:
        """Tavily API를 사용한 웹 검색"""
        print(f"==== [TAVILY WEB SEARCH: {query}] ====")
        
        self.search_stats["queries_processed"] += 1
        
        try:
            # 의료 검색에 최적화된 파라미터
            search_params = {
                "api_key": self.api_key,
                "query": self._optimize_medical_query(query),
                "max_results": max_results,
                "search_depth": "advanced",  # 심층 검색
                "include_domains": [
                    "pubmed.ncbi.nlm.nih.gov", "mayoclinic.org", 
                    "who.int", "cdc.gov", "nih.gov", "medlineplus.gov"
                ],
                "include_answer": True,
                "include_raw_content": True,
                "include_images": False
            }
            
            response = requests.post(self.api_url, json=search_params)
            response.raise_for_status()
            
            results = response.json()
            
            # Document 객체로 변환
            documents = self._convert_to_documents(results, query)
            
            self.search_stats["successful_searches"] += 1
            print(f"  ✅ Tavily 검색 완료: {len(documents)}개 결과")
            
            return documents
            
        except Exception as e:
            print(f"  ❌ Tavily 검색 실패: {str(e)}")
            self.search_stats["failed_searches"] += 1
            return self._create_fallback_documents(query)
    
    def _optimize_medical_query(self, query: str) -> str:
        """의료 검색을 위한 쿼리 최적화"""
        # 의료 관련 키워드 추가
        medical_terms = ["treatment", "symptoms", "diagnosis", "medical", "clinical", "healthcare"]
        
        # 기본 쿼리가 한글이면 영어 키워드 추가
        if any('\uAC00' <= char <= '\uD7A3' for char in query):  # 한글 감지
            query += " " + " ".join([term for term in medical_terms if term not in query.lower()])
        
        return query
    
    def _convert_to_documents(self, results: Dict, query: str) -> List[Document]:
        """검색 결과를 Document 객체로 변환"""
        documents = []
        
        # 생성된 답변 추가 (있는 경우)
        if "answer" in results and results["answer"]:
            answer_doc = Document(
                page_content=f"Tavily 생성 답변:\n\n{results['answer']}",
                metadata={
                    "source": "tavily_generated_answer",
                    "source_type": "web",
                    "query": query,
                    "generated_at": datetime.now().isoformat(),
                    "reliability": "medium",
                    "confidence": "medium"
                }
            )
            documents.append(answer_doc)
        
        # 검색 결과 추가
        for i, result in enumerate(results.get("results", [])):
            content = f"제목: {result.get('title', '제목 없음')}\n\n"
            
            # 본문 추가
            if "content" in result:
                content += f"내용: {result['content']}\n\n"
            elif "raw_content" in result:
                # 길이 제한
                raw_content = result["raw_content"][:1000] + "..." if len(result["raw_content"]) > 1000 else result["raw_content"]
                content += f"내용: {raw_content}\n\n"
            
            # URL 추가
            content += f"출처: {result.get('url', '알 수 없음')}"
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": result.get("url", f"tavily_result_{i}"),
                    "title": result.get("title", "제목 없음"),
                    "source_type": "web",
                    "query": query,
                    "rank": i + 1,
                    "score": result.get("score", 0),
                    "retrieved_at": datetime.now().isoformat(),
                    "reliability": "medium",
                    "domain": self._extract_domain(result.get("url", ""))
                }
            )
            documents.append(doc)
        
        return documents
    
    def _extract_domain(self, url: str) -> str:
        """URL에서 도메인 추출"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return "unknown"
    
    def _create_fallback_documents(self, query: str) -> List[Document]:
        """검색 실패 시 폴백 문서 생성"""
        return [Document(
            page_content=f"웹 검색 결과를 가져오는 중 오류가 발생했습니다. 다음 질문에 대해 로컬 문서를 참조하세요: {query}",
            metadata={
                "source": "tavily_fallback",
                "source_type": "web",
                "query": query,
                "error": True,
                "reliability": "low"
            }
        )]
        
    def get_stats(self) -> Dict[str, Any]:
        """검색 통계 반환"""
        success_rate = 0
        if self.search_stats["queries_processed"] > 0:
            success_rate = self.search_stats["successful_searches"] / self.search_stats["queries_processed"]
        
        return {
            "searcher_type": "TavilySearcher",
            "queries_processed": self.search_stats["queries_processed"],
            "successful_searches": self.search_stats["successful_searches"],
            "failed_searches": self.search_stats["failed_searches"],
            "success_rate": round(success_rate * 100, 2)
        }