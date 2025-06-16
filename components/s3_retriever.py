# components/s3_retriever.py
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
import boto3
import json
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)
boto3.setup_default_session(region_name='us-east-2')

class S3Retriever:
    """S3 기반 임베딩 검색 담당 클래스"""
    
    def __init__(self, 
                 bucket_name="aws-medical-chatbot",
                 search_function="arn:aws:lambda:us-east-2:481371222694:function:medical-embedding-search",
                 region_name="us-east-2", 
                 enabled=True):
        """
        S3 리트리버 초기화
        
        Args:
            bucket_name: S3 버킷 이름
            search_function: 임베딩 검색 Lambda 함수 이름
            enabled: S3 검색 활성화 여부
        """
        self.s3 = boto3.client('s3')
        self.lambda_client = boto3.client('lambda')
        self.bucket_name = bucket_name
        self.search_function = search_function
        self.enabled = enabled
        
        # 검색 통계
        self.search_stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "failed_searches": 0,
            "total_documents_found": 0,
            "average_response_time": 0.0
        }
        
        # 초기화 로그
        status = "활성화" if enabled else "비활성화"
        print(f"🔍 S3 리트리버 초기화 완료 (상태: {status})")
        print(f"   📦 버킷: {bucket_name}")
        print(f"   🔍 검색 함수: {search_function}")
    
    def retrieve_documents(self, question: str, k: int = 5, 
                          category_filter: Optional[str] = None, 
                          folder_filter: Optional[str] = None) -> List[Document]:
        """S3 임베딩 시스템에서 관련 문서 검색"""
        
        # 비활성화 상태면 빈 리스트 반환
        if not self.enabled:
            print("  ⚠️ S3 리트리버가 비활성화되어 있습니다.")
            return []
        
        print(f"==== [S3 SEARCH: {question[:50]}...] ====")
        start_time = time.time()
        
        try:
            # 검색 파라미터 구성
            payload = {
                'query': question,
                'top_k': k,
                'use_cache': True
            }
            
            # 필터 추가 (존재하는 경우만)
            if category_filter:
                payload['category'] = category_filter
                print(f"  🔍 카테고리 필터 적용: {category_filter}")
            
            if folder_filter:
                payload['folder'] = folder_filter
                print(f"  🔍 폴더 필터 적용: {folder_filter}")
            
            # Lambda 함수 호출
            print(f"  🔄 Lambda 함수 호출: {self.search_function}")
            response = self.lambda_client.invoke(
                FunctionName=self.search_function,
                InvocationType='RequestResponse',  # 동기 호출
                Payload=json.dumps(payload)
            )
            
            # 응답 처리
            payload_response = json.loads(response['Payload'].read().decode())
            
            if 'statusCode' in payload_response and payload_response['statusCode'] == 200:
                body = json.loads(payload_response['body'])
                results = body.get('results', [])
                
                # 검색 통계 업데이트
                self.search_stats["total_searches"] += 1
                self.search_stats["successful_searches"] += 1
                self.search_stats["total_documents_found"] += len(results)
                
                # 응답 시간 업데이트
                elapsed_time = time.time() - start_time
                self._update_average_response_time(elapsed_time)
                
                # 캐시 히트 여부
                from_cache = body.get('from_cache', False)
                cache_status = "캐시 사용" if from_cache else "신규 검색"
                
                # 결과 로깅
                print(f"  ✅ S3 검색 완료: {len(results)}개 문서 ({cache_status}, {elapsed_time:.2f}초)")
                
                # Document 객체로 변환
                documents = self._convert_to_documents(results)
                return documents
            else:
                # 오류 응답
                error_message = payload_response.get('body', '알 수 없는 오류')
                print(f"  ❌ S3 검색 실패: {error_message}")
                
                self.search_stats["total_searches"] += 1
                self.search_stats["failed_searches"] += 1
                
                return []
            
        except Exception as e:
            logger.error(f"S3 검색 실패: {str(e)}")
            print(f"  ❌ S3 검색 오류: {str(e)}")
            
            self.search_stats["total_searches"] += 1
            self.search_stats["failed_searches"] += 1
            
            return []
    
    def _convert_to_documents(self, search_results: List[Dict]) -> List[Document]:
        """검색 결과를 Document 객체로 변환"""
        documents = []
        
        for result in search_results:
            # Document 객체 생성
            doc = Document(
                page_content=result.get('text', '내용 없음'),
                metadata={
                    'source': f"s3://{self.bucket_name}/{result.get('text_path', '')}",
                    'document_id': result.get('document_id', ''),
                    'chunk_id': result.get('chunk_id', ''),
                    'similarity_score': result.get('similarity', 0.0),
                    'category': result.get('category', '일반의학'),
                    'page': result.get('page', 0),
                    'source_type': 's3'
                }
            )
            documents.append(doc)
        
        return documents
    
    def _update_average_response_time(self, elapsed_time: float) -> None:
        """평균 응답 시간 업데이트"""
        current_avg = self.search_stats["average_response_time"]
        successful_searches = self.search_stats["successful_searches"]
        
        if successful_searches <= 1:
            self.search_stats["average_response_time"] = elapsed_time
        else:
            # 이동 평균 계산
            new_avg = ((current_avg * (successful_searches - 1)) + elapsed_time) / successful_searches
            self.search_stats["average_response_time"] = new_avg
    
    def set_enabled(self, enabled: bool) -> None:
        """S3 리트리버 활성화/비활성화"""
        self.enabled = enabled
        status = "활성화" if enabled else "비활성화"
        print(f"🔧 S3 리트리버 상태 변경: {status}")
    
    def get_stats(self) -> Dict[str, Any]:
        """S3 리트리버 통계 반환"""
        success_rate = 0
        if self.search_stats["total_searches"] > 0:
            success_rate = self.search_stats["successful_searches"] / self.search_stats["total_searches"] * 100
        
        return {
            "retriever_type": "S3Retriever",
            "enabled": self.enabled,
            "bucket": self.bucket_name,
            "search_function": self.search_function,
            "total_searches": self.search_stats["total_searches"],
            "successful_searches": self.search_stats["successful_searches"],
            "failed_searches": self.search_stats["failed_searches"],
            "success_rate": f"{success_rate:.1f}%",
            "total_documents_found": self.search_stats["total_documents_found"],
            "average_response_time": f"{self.search_stats['average_response_time']:.2f}초"
        }