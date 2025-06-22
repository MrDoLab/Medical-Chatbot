# config.py
import os
from dotenv import load_dotenv
from typing import Dict, Any

# .env 파일 자동 로딩
load_dotenv()

class Config:
    """RAG 시스템 설정 클래스"""

    # 검색 소스 활성화 상태
    SEARCH_SOURCES_CONFIG = {
        "local": False,     # 로컬
        "s3": False,       # S3 
        "medgemma": False, # MedGemma 
        "pubmed": True,    # PubMed 
        "tavily": True,       # Tavily 웹 
        "bedrock_kb": True    # AWS Bedrock Knowledge Base 
    }

    # 검색 소스 가중치 
    SOURCE_WEIGHTS = {
            "pubmed": 1.0,      # 학술 논문
            "bedrock_kb": 0.95, # Bedrock Knowledge Base
            "local": 0.9,       # 로컬 문서
            "s3": 0.9,         # S3 저장 문서
            "medgemma": 0.8,    # 의료 특화 AI
            "tavily": 0.7       # 웹 검색
        }
    
    # LLM 설정
    MODEL_NAME = "gpt-4o"
    TEMPERATURE = 0
    
    # OpenAI 임베딩 설정
    EMBEDDING_CONFIG = {
        "model": "text-embedding-3-large",
        "dimensions": None,  # 기본 3072차원 사용 (최고 성능)
        "cache_enabled": True,
        "cache_ttl_days": 7,  # 캐시 유효 기간
        "batch_size": 100,  # 배치 처리 크기
        "max_retries": 3  # API 실패 시 재시도 횟수
    }
    
    # 검색 설정
    SEARCH_CONFIG = {
    "top_k": 5,  # 기본 검색 결과 개수
    "similarity_threshold": 0.3,  # 유사도 임계값
    "medical_keywords_boost": True,  # 의료 키워드 가중치 부여
    "category_filtering": True,  # 카테고리 기반 필터링
    "emergency_priority": True,  # 응급 문서 우선순위
    "enable_web_search": True,   # 웹 검색 활성화
    "reliable_domains": [        # 신뢰할 수 있는 의료 도메인
        "pubmed.ncbi.nlm.nih.gov", "mayoclinic.org", "who.int", 
        "cdc.gov", "nih.gov", "medlineplus.gov", "uptodate.com"
    ]
}
    
    # 유사도 임계값 (직접 접근용)
    SIMILARITY_THRESHOLD = 0.3
    
    # 벡터 스토어 설정 (레거시 - 사용 안함)
    VECTORSTORE_CONFIG = {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "persist_directory": "./chroma_db", 
        "search_kwargs": {"k": 3}
    }
    
    # 재작성 설정
    MAX_REWRITE_COUNT = 2  # 환각 체크 최대 재시도 횟수
    
    # 재귀 한도 설정  
    RECURSION_LIMIT = 50  # 간소화된 워크플로우로 줄임
    
    # 의료 도메인 특화 설정
    MEDICAL_CONFIG = {
        "categories": [
            "응급처치", "내과", "외과", "소아과", "산부인과", 
            "정신과", "영상의학", "병리학", "약학", "간호학",
            "만성질환관리", "약물정보", "진단검사", "감염관리"
        ],
        "severity_levels": ["low", "medium", "high", "critical"],
        "departments": [
            "응급의학과", "내과", "외과", "소아과", "산부인과",
            "정형외과", "신경과", "정신건강의학과", "영상의학과"
        ]
    }

    # MedGemma 설정
    MEDGEMMA_CONFIG = {
        "model_name": "google/gemma-2b-it",  # 또는 사용 중인 모델
        "device": "auto",
        "max_response_length": 1024,  # 충분히 긴 응답 허용
        "min_response_length": 100,   # 최소 응답 길이 지정
        "enable_medgemma": False,
        "fallback_on_error": True,
        "generation_params": {
            "temperature": 0.8,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            "do_sample": True,
            "num_return_sequences": 1,
            "num_beams": 1,
            "early_stopping": True
        },
        "memory_optimization": {
            "load_in_8bit": False,
            "load_in_4bit": True,
            "use_flash_attention": True,
            "low_cpu_mem_usage": True
        }
    }
    
    # AWS Bedrock 설정
    BEDROCK_CONFIG = {
        "kb_id": "IZJR1RYKEY",  # 실제 KB ID로 변경
        "region": "us-east-2",  # 실제 리전으로 변경
        "enabled": True,  # Bedrock 검색 활성화 여부
        "confidence_threshold": 0.3  # 최소 신뢰도 임계값
    }
    

    # S3 임베딩 설정 추가
    S3_CONFIG = {
        "bucket_name": "aws-medical-chatbot",
        "search_function": "medical-embedding-search",
        "enabled": True,  # 기본 활성화
        "cache_ttl_days": 7,  # 캐시 유효 기간
        "max_retries": 3  # API 실패 시 재시도 횟수
    }

    # 시스템 프롬프트들 (의료 특화)   

    @classmethod
    def get_medical_categories(cls) -> list:
        """의료 카테고리 목록 반환"""
        return cls.MEDICAL_CONFIG["categories"]
    
    @classmethod  
    def get_embedding_model(cls) -> str:
        """현재 임베딩 모델명 반환"""
        return cls.EMBEDDING_CONFIG["model"]
    
    @classmethod
    def is_openai_embedding(cls) -> bool:
        """OpenAI 임베딩 사용 여부"""
        return "text-embedding" in cls.EMBEDDING_CONFIG["model"]
    
    @classmethod
    def get_search_threshold(cls) -> float:
        """검색 유사도 임계값"""
        return cls.SIMILARITY_THRESHOLD
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """설정 유효성 검증"""
        issues = []
        
        # 필수 환경변수 확인
        import os
        if not os.getenv("OPENAI_API_KEY"):
            issues.append("OPENAI_API_KEY 환경변수가 설정되지 않음")
        
        # 모델명 검증
        if not cls.EMBEDDING_CONFIG["model"].startswith("text-embedding"):
            issues.append("지원되지 않는 임베딩 모델")
        
        # 임계값 범위 확인
        threshold = cls.SIMILARITY_THRESHOLD
        if not 0.0 <= threshold <= 1.0:
            issues.append("유사도 임계값이 범위를 벗어남 (0.0-1.0)")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "config_type": "refactored_system",
            "embedding_model": cls.EMBEDDING_CONFIG["model"],
            "similarity_threshold": cls.SIMILARITY_THRESHOLD,
            "max_retries": cls.MAX_REWRITE_COUNT
        }
    
    @classmethod
    def update_system_prompt(cls, prompt_type: str, new_content: str) -> bool:
        """시스템 프롬프트 업데이트"""
        try:
            if hasattr(cls, prompt_type):
                setattr(cls, prompt_type, new_content)
                
                # prompts.py의 SystemPrompts도 업데이트
                try:
                    from prompts import SystemPrompts
                    system_prompts = SystemPrompts()
                    
                    # 프롬프트 이름 변환 (Config 형식 -> prompts.py 형식)
                    prompt_map = {
                        "RAG_SYSTEM_PROMPT": "RAG",
                        "ROUTER_SYSTEM_PROMPT": "ROUTER",
                        "GRADER_SYSTEM_PROMPT": "GRADER",
                        "HALLUCINATION_SYSTEM_PROMPT": "HALLUCINATION",
                        "REWRITER_SYSTEM_PROMPT": "REWRITER"
                    }
                    
                    if prompt_type in prompt_map:
                        mapped_name = prompt_map[prompt_type]
                        system_prompts.update(mapped_name, new_content)
                except Exception as e:
                    print(f"SystemPrompts 업데이트 실패: {e}")
                
                return True
            return False
        except Exception as e:
            print(f"프롬프트 업데이트 실패: {e}")
            return False

    @classmethod
    def get_system_prompt(cls, prompt_name: str, **kwargs):
        """
        prompts.py의 시스템 프롬프트를 가져옵니다.
        
        Args:
            prompt_name: 프롬프트 이름 (예: 'RAG', 'GRADER')
            kwargs: 템플릿 변수
            
        Returns:
            프롬프트 내용
        """
        from prompts import system_prompts
        return system_prompts.format(prompt_name, **kwargs) if kwargs else system_prompts.get(prompt_name)