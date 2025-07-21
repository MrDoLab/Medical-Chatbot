# components/parallel_searcher.py
"""
병렬 검색 전용 클래스 - 다중 소스 동시 검색 관리
"""

from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.documents import Document
import logging

from config import Config

logger = logging.getLogger(__name__)

class ParallelSearcher:
    """다중 소스 병렬 검색 관리자"""
    
    def __init__(self, 
                 local_retriever=None, 
                 medgemma_searcher=None,
                 pubmed_searcher=None,
                 tavily_searcher=None, 
                 s3_retriever=None, 
                 bedrock_retriever=None):
        """
        병렬 검색기 초기화
        
        Args:
            local_retriever: 로컬 문서 검색기
            s3_retriever: S3 기반 검색기
            medgemma_searcher: MedGemma 검색기
            pubmed_searcher: PubMed 검색기
            tavily_searcher: Tavily 웹 검색기
            bedrock_retriever: Bedrock KB 검색기
        """

        self.retrievers = {
            "local": local_retriever,
            "s3": s3_retriever,
            "medgemma": medgemma_searcher,
            "pubmed": pubmed_searcher,
            "tavily": tavily_searcher,
            "bedrock_kb": bedrock_retriever
        }
                    
        # 소스별 활성화 상태
        self.sources_enabled = {
            source: retriever is not None and Config.SEARCH_SOURCES_CONFIG.get(source, False)
            for source, retriever in self.retrievers.items()
        }

        # 병렬 실행 설정
        self.max_workers = 6
        self.timeout = 120  # 각 소스별 타임아웃 (초)
        
        # 활성화된 소스 로깅
        active_sources = [source for source, enabled in self.sources_enabled.items() if enabled]
        print(f"🚀 병렬 검색기 초기화 완료 (활성 소스: {', '.join(active_sources) if active_sources else '없음'})")

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
        source_functions = {}
        
        # 로컬 검색기 작업 추가
        if self.retrievers["local"] is not None:
            try:
                if hasattr(self.retrievers["local"], "_retrieve_local_documents"):
                    func = self.retrievers["local"]._retrieve_local_documents
                else:
                    func = self.retrievers["local"].retrieve_documents
                    
                source_functions["local"] = {
                    "function": func,
                    "args": [question],
                    "kwargs": {}
                }
            except Exception as e:
                print(f"  ⚠️ 로컬 검색기 설정 실패: {str(e)}")
        
        # S3 검색기 작업 추가
        if self.retrievers["s3"] is not None:
            try:
                source_functions["s3"] = {
                    "function": self.retrievers["s3"].retrieve_documents,
                    "args": [question],
                    "kwargs": {}
                }
            except Exception as e:
                print(f"  ⚠️ S3 검색기 설정 실패: {str(e)}")
        
        # MedGemma 검색기 작업 추가
        if self.retrievers["medgemma"] is not None:
            try:
                source_functions["medgemma"] = {
                    "function": self.retrievers["medgemma"].search_medgemma,
                    "args": [question],
                    "kwargs": {"max_results": 3}
                }
            except Exception as e:
                print(f"  ⚠️ MedGemma 검색기 설정 실패: {str(e)}")
        
        # PubMed 검색기 작업 추가
        if self.retrievers["pubmed"] is not None:
            try:
                source_functions["pubmed"] = {
                    "function": self.retrievers["pubmed"].search_pubmed,
                    "args": [question],
                    "kwargs": {"max_results": 3}
                }
            except Exception as e:
                print(f"  ⚠️ PubMed 검색기 설정 실패: {str(e)}")
        
        # Tavily 검색기 작업 추가
        if self.retrievers["tavily"] is not None:
            try:
                source_functions["tavily"] = {
                    "function": self.retrievers["tavily"].search_web,
                    "args": [question],
                    "kwargs": {"max_results": 5}
                }
            except Exception as e:
                print(f"  ⚠️ Tavily 검색기 설정 실패: {str(e)}")
        
        # Bedrock KB 검색기 작업 추가
        if self.retrievers["bedrock_kb"] is not None:
            try:
                source_functions["bedrock_kb"] = {
                    "function": self.retrievers["bedrock_kb"].retrieve_documents,
                    "args": [question],
                    "kwargs": {}
                }
            except Exception as e:
                print(f"  ⚠️ Bedrock KB 검색기 설정 실패: {str(e)}")
        
        # 활성화된 소스만 작업에 추가
        for source, enabled in self.sources_enabled.items():
            if enabled and source in source_functions:
                print(f"  🔍 {source.upper()} 검색 추가")
                tasks[source] = source_functions[source]
        
        # 작업이 없는 경우 로깅
        if not tasks:
            print("  ⚠️ 활성화된 검색 소스가 없거나 모두 초기화 실패했습니다")
        
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
        if source == "local" and self.retrievers.get("local"):
            self.retrievers["local"].set_local_search_enabled(enabled)
        elif source == "s3" and self.retrievers.get("s3"):
            self.retrievers["s3"].set_enabled(enabled)
        
        # 상태 업데이트
        self.sources_enabled[source] = enabled
        
        status = "활성화" if enabled else "비활성화"
        print(f"🔧 검색 소스 '{source}' {status} 완료")

    def get_stats(self) -> Dict[str, Any]:
        """병렬 검색기 통계 반환"""
        stats = {
            "active_sources": [source for source, enabled in self.sources_enabled.items() if enabled],
            "total_sources": len(self.retrievers),
            "enabled_sources": sum(1 for enabled in self.sources_enabled.values() if enabled)
        }
        
        # 각 검색기의 통계도 추가
        for source, retriever in self.retrievers.items():
            if retriever and hasattr(retriever, "get_stats"):
                try:
                    stats[f"{source}_stats"] = retriever.get_stats()
                except:
                    pass
        
        return stats    

    def _normalize_metadata(self, documents: List[Document], source_type: str) -> List[Document]:
        """메타데이터 표준화 및 URL 확인"""
        if not documents:
            return []
        
        normalized_docs = []
        
        for doc in documents:
            # 복사본 생성
            new_doc = Document(
                page_content=doc.page_content,
                metadata=doc.metadata.copy()
            )
            
            # source_type이 없는 경우 추가
            if "source_type" not in new_doc.metadata:
                new_doc.metadata["source_type"] = source_type
            
            # URL 정보 확인 및 정규화
            self._ensure_url_in_metadata(new_doc.metadata, source_type)
            
            normalized_docs.append(new_doc)
        
        return normalized_docs

    def _ensure_url_in_metadata(self, metadata: Dict[str, Any], source_type: str) -> None:
        """메타데이터에 URL 정보가 있는지 확인하고 없으면 생성"""
        # 이미 URL이 있는 경우 건너뜀
        if "url" in metadata and metadata["url"]:
            return
        
        # 소스 타입별 URL 생성 로직
        if source_type == "pubmed":
            # PubMed ID 기반 URL 생성
            pmid = metadata.get("pmid")
            if pmid:
                metadata["url"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        
        elif source_type == "bedrock_kb":
            # S3 위치 정보 기반 URL 생성
            s3_location = metadata.get("s3_location")
            if s3_location and s3_location.startswith("s3://"):
                parts = s3_location[5:].split("/", 1)
                if len(parts) == 2:
                    bucket, key = parts
                    region = "us-east-1"  # 기본 리전 (실제 환경에 맞게 조정 필요)
                    metadata["url"] = f"https://s3.console.aws.amazon.com/s3/object/{bucket}?region={region}&prefix={key}"
        
        elif source_type == "medgemma":
            # MedGemma 모델 카드 URL 생성
            model_name = metadata.get("model_name", "google/medgemma_27b_text_it")
            if model_name:
                metadata["url"] = f"https://huggingface.co/{model_name}"
        
        elif source_type == "local":
            # 로컬 문서는 파일 경로를 URL로 간주
            source = metadata.get("source", "")
            if source:
                metadata["url"] = f"file://{source}"
        
        # source가 이미 URL인 경우
        source = metadata.get("source", "")
        if not metadata.get("url") and (source.startswith("http://") or source.startswith("https://")):
            metadata["url"] = source