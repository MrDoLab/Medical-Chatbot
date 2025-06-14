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
    
    def __init__(self, retriever, medgemma_searcher=None):
        """
        병렬 검색기 초기화
        
        Args:
            retriever: RAG 검색 담당 (임베딩 기반)
            medgemma_searcher: MedGemma 검색 담당 (optional)
        """
        self.retriever = retriever
        self.medgemma_searcher = medgemma_searcher
        
        # 병렬 실행 설정
        self.max_workers = 3
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
        
        # RAG 검색 (항상 사용 가능)
        tasks["rag"] = {
            "function": self.retriever.retrieve_documents,
            "args": [question],
            "kwargs": {}
        }
        
        # PubMed 검색 (사용 가능한 경우)
        if hasattr(self.retriever, 'pubmed_searcher') and self.retriever.pubmed_searcher:
            tasks["pubmed"] = {
                "function": self.retriever.pubmed_searcher.search_pubmed,
                "args": [question],
                "kwargs": {"max_results": 3}
            }
        
        # MedGemma 검색 (사용 가능한 경우)
        if self.medgemma_searcher is not None:
            tasks["medgemma"] = {
                "function": self.medgemma_searcher.search_medgemma,
                "args": [question],
                "kwargs": {"max_results": 1}
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