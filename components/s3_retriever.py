# components/s3_retriever.py
from typing import List, Dict, Optional
from langchain_core.documents import Document
import boto3
import json
import logging

logger = logging.getLogger(__name__)

# 상수 정의
DEFAULT_REGION = 'us-east-2'
DEFAULT_BUCKET = 'aws-medical-chatbot'
DEFAULT_FUNCTION = 'arn:aws:lambda:us-east-2:481371222694:function:medical-embedding-search'

class S3Retriever:
    """S3 기반 임베딩 검색 담당 클래스"""
    
    def __init__(self, 
                 bucket_name=DEFAULT_BUCKET,
                 search_function=DEFAULT_FUNCTION,
                 region_name=DEFAULT_REGION, 
                 enabled=True):
        """S3 리트리버 초기화"""
        boto3.setup_default_session(region_name=region_name)
        self.s3 = boto3.client('s3')
        self.lambda_client = boto3.client('lambda')
        self.bucket_name = bucket_name
        self.search_function = search_function
        self.enabled = enabled
        
        logger.info(f"S3 리트리버 초기화 완료 (상태: {'활성화' if enabled else '비활성화'})")
    
    def retrieve_documents(self, question: str, k: int = 5, 
                          category_filter: Optional[str] = None, 
                          folder_filter: Optional[str] = None) -> List[Document]:
        """S3 임베딩 시스템에서 관련 문서 검색"""
        # 비활성화 상태면 빈 리스트 반환
        if not self.enabled:
            logger.info("S3 리트리버가 비활성화됨")
            return []
        
        logger.info(f"S3 검색 실행: {question[:50]}...")
        
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
            
            if folder_filter:
                payload['folder'] = folder_filter
            
            # Lambda 함수 호출
            response = self.lambda_client.invoke(
                FunctionName=self.search_function,
                InvocationType='RequestResponse',
                Payload=json.dumps(payload),
                Timeout=30  # 30초 타임아웃 추가
            )
            
            # 응답 처리
            payload_response = json.loads(response['Payload'].read().decode())
            
            if 'statusCode' in payload_response and payload_response['statusCode'] == 200:
                body = json.loads(payload_response['body'])
                results = body.get('results', [])
                logger.info(f"S3 검색 완료: {len(results)}개 문서")
                
                # Document 객체로 변환
                documents = self._convert_to_documents(results)
                return documents
            else:
                # 오류 응답
                error_message = payload_response.get('body', '알 수 없는 오류')
                logger.error(f"S3 검색 실패: {error_message}")
                return []
            
        except Exception as e:
            logger.error(f"S3 검색 오류: {str(e)}")
            return []
    
    def _convert_to_documents(self, search_results: List[Dict]) -> List[Document]:
        """검색 결과를 Document 객체로 변환"""
        documents = []
        
        for result in search_results:
            # 간소화된 메타데이터로 Document 생성
            doc = Document(
                page_content=result.get('text', '내용 없음'),
                metadata={
                    'source': f"s3://{self.bucket_name}/{result.get('text_path', '')}",
                    'similarity_score': result.get('similarity', 0.0),
                    'category': result.get('category', '일반의학'),
                    'source_type': 's3'
                }
            )
            documents.append(doc)
        
        return documents
    
    def set_enabled(self, enabled: bool) -> None:
        """S3 리트리버 활성화/비활성화"""
        self.enabled = enabled
        logger.info(f"S3 리트리버 상태 변경: {'활성화' if enabled else '비활성화'}")