import boto3
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid

class QuestionLogger:
    """의료 챗봇 질문 로깅 담당 클래스"""
    
    def __init__(self, queue_url: str = None, region: str = 'ap-northeast-2'):
        """로거 초기화"""
        self.sqs = boto3.client('sqs', region_name=region)
        
        # 기본 큐 URL 설정
        self.queue_url = queue_url or "https://sqs.[region].amazonaws.com/[account-id]/medical-questions-queue"
        
        # 로깅 통계
        self.stats = {
            "questions_logged": 0,
            "log_failures": 0
        }
        
        print("📝 질문 로거 초기화 완료")
    
    def log_question(self, 
                    question: str, 
                    user_id: str = "anonymous", 
                    session_id: Optional[str] = None,
                    answer: Optional[str] = None,
                    sources_used: Optional[List[str]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """질문 데이터를 SQS에 로깅"""
        try:
            # 기본값 설정
            session_id = session_id or str(uuid.uuid4())
            current_time = datetime.now().isoformat()
            
            # 메시지 구성
            message = {
                'question': question,
                'user_id': user_id,
                'session_id': session_id,
                'timestamp': current_time,
                'sources_used': sources_used or [],
                'metadata': metadata or {}
            }
            
            # 답변이 있으면 추가
            if answer:
                message['answer'] = answer
                message['answer_timestamp'] = current_time
            
            # SQS에 메시지 전송
            response = self.sqs.send_message(
                QueueUrl=self.queue_url,
                MessageBody=json.dumps(message)
            )
            
            # 통계 업데이트
            self.stats["questions_logged"] += 1
            
            print(f"✅ 질문 로깅 성공: {response.get('MessageId')}")
            return True
            
        except Exception as e:
            print(f"❌ 질문 로깅 실패: {str(e)}")
            self.stats["log_failures"] += 1
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """로깅 통계 반환"""
        return {
            "logger_type": "QuestionLogger",
            "queue_url": self.queue_url,
            "questions_logged": self.stats["questions_logged"],
            "log_failures": self.stats["log_failures"],
            "success_rate": self._calculate_success_rate()
        }
    
    def _calculate_success_rate(self) -> float:
        """성공률 계산"""
        total = self.stats["questions_logged"] + self.stats["log_failures"]
        if total == 0:
            return 1.0
        return self.stats["questions_logged"] / total