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
                doc = Document(
                    page_content=result['content']['text'],
                    metadata={
                        "source": "bedrock_kb",
                        "score": float(result['score']),
                        "s3_location": result['location']['s3Location']['uri'] if 's3Location' in result['location'] else "unknown"
                    }
                )
                documents.append(doc)
            
            print(f"  ✅ Bedrock 검색 완료: {len(documents)}개 문서")
            return documents
        except Exception as e:
            print(f"  ❌ Bedrock 검색 실패: {str(e)}")
            return []