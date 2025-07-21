# components/tavily_searcher.py
from typing import List, Dict
from langchain_core.documents import Document
import requests
from datetime import datetime
import os
from urllib.parse import urlparse

class TavilySearcher:
    """Tavily API 기반 웹 검색 담당 클래스"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.api_url = "https://api.tavily.com/search"
        
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY가 설정되지 않았습니다!")
    
    def search_web(self, query: str, max_results: int = 5) -> List[Document]:
        """Tavily API를 사용한 웹 검색"""
        try:
            # 한글 쿼리인 경우 영어 의료 키워드 추가
            optimized_query = query
            if any('\uAC00' <= char <= '\uD7A3' for char in query):
                optimized_query += " medical clinical treatment diagnosis"
            
            # 검색 요청
            search_params = {
                "api_key": self.api_key,
                "query": optimized_query,
                "max_results": max_results,
                "search_depth": "advanced",
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
            documents = []
            
            # 생성된 답변 추가 (있는 경우)
            if "answer" in results and results["answer"]:
                # 참조한 출처 URL 목록 생성
                citation_urls = [result.get("url", "") for result in results.get("results", []) if "url" in result]
                citation_info = "\n".join([f"- {url}" for url in citation_urls[:3]]) if citation_urls else "별도 인용 정보 없음"
                
                documents.append(Document(
                    page_content=f"Tavily 생성 답변:\n\n{results['answer']}\n\n참조 출처:\n{citation_info}",
                    metadata={
                        "source": "tavily_generated_answer",
                        "source_type": "tavily",
                        "query": query,
                        "generated_at": datetime.now().isoformat(),
                        "citation_urls": citation_urls,
                        "url": "https://tavily.com/",  # 기본 Tavily 웹사이트
                        "title": "Tavily AI 생성 답변"
                    }
                ))
            
            # 검색 결과 추가
            for i, result in enumerate(results.get("results", [])):
                url = result.get("url", "")
                domain = urlparse(url).netloc if url else "unknown-domain"
                
                content = f"제목: {result.get('title', '제목 없음')}\n\n"
                
                # 본문 추가 (raw_content 우선)
                if "raw_content" in result:
                    content_text = result["raw_content"][:1000] + "..." if len(result["raw_content"]) > 1000 else result["raw_content"]
                    content += f"내용: {content_text}\n\n"
                elif "content" in result:
                    content += f"내용: {result['content']}\n\n"
                
                content += f"출처: {url}"
                
                # 도메인 신뢰도 평가
                reliability = "high" if any(trusted in domain for trusted in ["nih.gov", "who.int", "cdc.gov", "mayoclinic.org", "pubmed"]) else "medium"
                
                # 발행일 추출 (있는 경우)
                published_date = result.get("published_date", "")
                
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": url,
                        "source_type": "tavily",
                        "title": result.get("title", "제목 없음"),
                        "url": url,  # 명시적으로 url 필드 추가
                        "domain": domain,
                        "rank": i + 1,
                        "reliability": reliability,
                        "published_date": published_date,
                        "retrieved_at": datetime.now().isoformat(),
                        "search_engine": "tavily"
                    }
                ))
            
            return documents
            
        except Exception as e:
            # 오류 발생 시 폴백 문서 반환
            return [Document(
                page_content=f"웹 검색 결과를 가져오는 중 오류가 발생했습니다. 다음 질문에 대해 로컬 문서를 참조하세요: {query}",
                metadata={
                    "source": "tavily_fallback",
                    "source_type": "tavily",
                    "error": True,
                    "error_message": str(e),
                    "url": "https://tavily.com/",
                    "retrieved_at": datetime.now().isoformat()
                }
            )]