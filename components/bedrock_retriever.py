# components/bedrock_retriever.py
from typing import List
from langchain_core.documents import Document
import boto3

class BedrockRetriever:
    def __init__(self, kb_id=None, region="us-east-1"):
        self.kb_id = kb_id
        self.region = region
        self.bedrock_agent = boto3.client('bedrock-agent-runtime', region_name=region)
        print(f"🔍 Bedrock Retriever 초기화 완료 (KB_ID: {kb_id})")
    
    def retrieve_documents(self, query, top_k=5):
        try:
            response = self.bedrock_agent.retrieve(
                knowledgeBaseId=self.kb_id,
                retrievalQuery={'text': query},
                retrievalConfiguration={
                    'vectorSearchConfiguration': {'numberOfResults': top_k}
                }
            )
            
            documents = []
            for result in response.get('retrievalResults', []):
                # S3 위치 정보 처리
                s3_location = result['location'].get('s3Location', {}).get('uri', "unknown")
                
                # 문서 제목 추출 (파일명에서)
                title = self._extract_title_from_s3_uri(s3_location)
                
                # 관리 콘솔 URL 생성 (직접 접근 대신 콘솔 링크 제공)
                console_url = self._create_s3_console_url(s3_location)
                
                # 문서 타입 식별
                doc_type = self._determine_document_type(s3_location)
                
                # 메타데이터 구성
                metadata = {
                    "source": "bedrock_kb",
                    "source_type": "bedrock_kb", 
                    "score": float(result['score']),
                    "s3_location": s3_location,
                    "console_url": console_url,  
                    "title": title,
                    "doc_type": doc_type,
                    "url": console_url,  
                    "kb_id": self.kb_id  
                }
                
                doc = Document(
                    page_content=result['content']['text'],
                    metadata=metadata
                )
                documents.append(doc)                
            
            print(f"  ✅ Bedrock 검색 완료: {len(documents)}개 문서")
            return documents
        except Exception as e:
            print(f"  ❌ Bedrock 검색 실패: {str(e)}")
            return []

    def _extract_title_from_s3_uri(self, s3_uri: str) -> str:
        """S3 URI에서 문서 제목 추출"""
        if not s3_uri or s3_uri == "unknown":
            return "Unknown Document"
        
        try:
            # s3://bucket-name/path/to/document.pdf 형식에서 파일명 추출
            parsed = urlparse(s3_uri)
            path = parsed.path if parsed.path else parsed.netloc
            
            # 경로에서 파일명 추출
            filename = path.split('/')[-1]
            
            # 파일 확장자 제거 및 언더스코어 대체
            title = re.sub(r'\.[^.]+$', '', filename)
            title = title.replace('_', ' ').replace('-', ' ')
            
            # 타이틀 포맷팅 (첫 글자 대문자화)
            title = ' '.join(word.capitalize() for word in title.split())
            
            return title or "Untitled Document"
        except:
            return "Untitled Document"
    
    def _create_s3_console_url(self, s3_uri: str) -> str:
        """S3 URI를 AWS 콘솔 URL로 변환"""
        if not s3_uri or s3_uri == "unknown":
            return ""
        
        try:
            # s3://bucket-name/key 형식 파싱
            if s3_uri.startswith('s3://'):
                parts = s3_uri[5:].split('/', 1)
                if len(parts) == 2:
                    bucket, key = parts
                    # AWS 콘솔 URL 생성
                    return f"https://s3.console.aws.amazon.com/s3/object/{bucket}?region={self.region}&prefix={key}"
            
            # 파싱 실패시 원래 URI 반환
            return s3_uri
        except:
            return s3_uri
    
    def _determine_document_type(self, s3_uri: str) -> str:
        """파일 확장자 기반으로 문서 타입 결정"""
        if not s3_uri or s3_uri == "unknown":
            return "unknown"
        
        lower_uri = s3_uri.lower()
        if lower_uri.endswith('.pdf'):
            return "pdf"
        elif lower_uri.endswith('.docx') or lower_uri.endswith('.doc'):
            return "word"
        elif lower_uri.endswith('.xlsx') or lower_uri.endswith('.xls'):
            return "excel"
        elif lower_uri.endswith('.txt'):
            return "text"
        elif lower_uri.endswith('.md'):
            return "markdown"
        elif lower_uri.endswith('.json'):
            return "json"
        else:
            return "document"