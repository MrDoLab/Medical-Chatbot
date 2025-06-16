# components/parallel_searcher.py
"""
병렬 검색 전용 클래스 - 다중 소스 동시 검색 관리
RAG + PubMed + MedGemma 를 진짜 병렬로 실행
"""

from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

class ParallelSearcher:
    """다중 소스 병렬 검색 관리자"""
    
    def __init__(self, retriever, medgemma_searcher=None, tavily_searcher=None, s3_retriever=None):
        """
        병렬 검색기 초기화
        
        Args:
            retriever: RAG 검색 담당 (임베딩 기반)
            tavliy_searcher : Web 검색 담당
            medgemma_searcher: MedGemma 검색 담당 (optional)
            s3_retriever: S3 임베딩 검색 담당 (optional)
        """
        self.retriever = retriever
        self.medgemma_searcher = medgemma_searcher
        self.tavily_searcher = tavily_searcher
        self.s3_retriever = s3_retriever
        
        # 소스별 활성화 상태
        self.sources_enabled = {
            "rag": retriever and retriever.local_search_enabled,
            "medgemma": medgemma_searcher is not None,
            "s3": s3_retriever is not None and s3_retriever.enabled,
            "pubmed": retriever and hasattr(retriever, 'pubmed_searcher') and retriever.pubmed_searcher is not None
        }

        # 병렬 실행 설정
        self.max_workers = 4
        self.timeout = 30  # 각 소스별 타임아웃 (초)
        
        print("🚀 병렬 검색기 초기화 완료")
    
    def search_all_parallel(self, question: str) -> Dict[str, List[Document]]:
        """
        모든 소스에서 병렬 검색 실행
        
        Args:
            question: 검색 질문
        
        Returns:
            소스별 검색 결과 딕셔너리
        """
        print(f"==== [PARALLEL SEARCH: {question[:50]}...] ====")
        
        # 검색 작업 준비
        search_tasks = self._prepare_search_tasks(question)
        
        if not search_tasks:
            print("  ❌ 사용 가능한 검색 소스가 없습니다")
            return {}
        
        # 병렬 실행
        results = self._execute_parallel_search(search_tasks)
        
        # 결과 로깅
        total_docs = sum(len(docs) for docs in results.values())
        successful_sources = len([k for k, v in results.items() if v])
        print(f"  📊 병렬 검색 완료: {successful_sources}/{len(search_tasks)}개 소스, {total_docs}개 문서")
        
        return results
    
    def _prepare_search_tasks(self, question: str) -> Dict[str, Dict]:
        """검색 작업 딕셔너리 준비"""
        tasks = {}
        
        # 로컬 RAG 검색 (활성화된 경우)
        if self.sources_enabled["rag"]:
            tasks["rag"] = {
                "function": self.retriever._retrieve_local_documents,
                "args": [question],
                "kwargs": {}
            }
        
        # PubMed 검색 (활성화된 경우)
        if hasattr(self.retriever, 'pubmed_searcher') and self.retriever.pubmed_searcher:
            tasks["pubmed"] = {
                "function": self.retriever.pubmed_searcher.search_pubmed,
                "args": [question],
                "kwargs": {"max_results": 3}
            }
        
        # MedGemma 검색 (활성화된 경우)
        if self.sources_enabled["pubmed"]:
            tasks["medgemma"] = {
                "function": self.medgemma_searcher.search_medgemma,
                "args": [question],
                "kwargs": {"max_results": 1}
            }
        
        # Tavily 웹 검색 (활성화된 경우)
        if self.sources_enabled["medgemma"]:
            tasks["web"] = {
                "function": self.tavily_searcher.search_web,
                "args": [question],
                "kwargs": {"max_results": 3}
            }

        # S3 검색 (활성화된 경우)
        if self.sources_enabled["s3"]:
            tasks["s3"] = {
                "function": self.s3_retriever.retrieve_documents,
                "args": [question],
                "kwargs": {"k": 5}
            }
        
        return tasks
    
    def _execute_parallel_search(self, search_tasks: Dict[str, Dict]) -> Dict[str, List[Document]]:
        """병렬 검색 실행"""
        results = {}
        
        print(f"  🔄 {len(search_tasks)}개 소스 병렬 검색 시작...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 모든 검색 작업 제출
            future_to_source = {}
            
            for source, task_info in search_tasks.items():
                try:
                    future = executor.submit(
                        task_info["function"],
                        *task_info["args"],
                        **task_info["kwargs"]
                    )
                    future_to_source[future] = source
                except Exception as e:
                    print(f"    ❌ {source} 작업 제출 실패: {str(e)}")
                    results[source] = []
            
            # 결과 수집 (타임아웃 적용)
            for future in as_completed(future_to_source, timeout=self.timeout + 5):
                source = future_to_source[future]
                
                try:
                    result = future.result(timeout=self.timeout)
                    results[source] = result if result else []
                    print(f"    ✅ {source}: {len(results[source])}개 문서")
                    
                except Exception as e:
                    print(f"    ❌ {source}: 검색 실패 - {str(e)}")
                    results[source] = []
        
        # 실행되지 않은 소스들 기본값 설정
        for source in search_tasks.keys():
            if source not in results:
                results[source] = []
        
        return results

    def set_source_enabled(self, source: str, enabled: bool) -> None:
        """특정 검색 소스 활성화/비활성화"""
        if source not in self.sources_enabled:
            print(f"⚠️ 알 수 없는 소스: {source}")
            return
        
        # 소스별 특수 처리
        if source == "rag" and self.retriever:
            self.retriever.set_local_search_enabled(enabled)
        elif source == "s3" and self.s3_retriever:
            self.s3_retriever.set_enabled(enabled)
        
        # 상태 업데이트
        self.sources_enabled[source] = enabled
        
        status = "활성화" if enabled else "비활성화"
        print(f"🔧 검색 소스 '{source}' {status} 완료")