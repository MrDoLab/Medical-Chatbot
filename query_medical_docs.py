import boto3
import json

# Knowledge Base ID 확인
# (AWS Bedrock 콘솔에서 Knowledge Base 세부 정보에서 확인 가능)
KNOWLEDGE_BASE_ID = "IZJR1RYKEY"  # 실제 ID로 변경하세요
REGION = "us-east-2"  # 실제 리전으로 변경하세요

# Bedrock Agent Runtime 클라이언트 생성
bedrock_agent = boto3.client('bedrock-agent-runtime', region_name=REGION)

def query_medical_docs(query_text):
    """의료 문서에 대한 쿼리 실행"""
    try:
        response = bedrock_agent.retrieve(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            retrievalQuery={
                'text': query_text
            },
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': 5
                }
            }
        )
        
        # 결과 처리
        results = response['retrievalResults']
        
        print(f"\n질문: {query_text}")
        print("\n검색 결과:")
        
        for i, result in enumerate(results, 1):
            print(f"\n--- 결과 {i} ---")
            print(f"문서: {result['location']['s3Location']['uri']}")
            print(f"점수: {result['score']}")
            print(f"내용 미리보기: {result['content']['text'][:200]}...")
        
        return results
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return None

# 대화형 인터페이스
if __name__ == "__main__":
    print("의료 문서 RAG 시스템")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
    
    while True:
        query = input("\n질문을 입력하세요: ")
        
        if query.lower() in ['quit', 'exit', '종료']:
            print("프로그램을 종료합니다.")
            break
        
        if query.strip():
            query_medical_docs(query)